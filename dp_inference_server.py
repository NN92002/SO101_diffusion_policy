#!/usr/bin/env python3

import argparse
import pickle
import socket
import struct
import time

import hydra
import numpy as np
import torch


DEFAULT_CKPT = (
    "/home/itri2026-3090/dp_checkpoints/"
    "rgb_20260714/best.ckpt"
)


def recv_exact(conn, n_bytes):
    data = b""

    while len(data) < n_bytes:
        packet = conn.recv(n_bytes - len(data))

        if not packet:
            return None

        data += packet

    return data


def recv_msg(conn):
    raw_len = recv_exact(conn, 4)

    if raw_len is None:
        return None

    msg_len = struct.unpack(">I", raw_len)[0]
    data = recv_exact(conn, msg_len)

    if data is None:
        return None

    return pickle.loads(data)


def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">I", len(data)) + data)


class DPInferenceServer:

    def __init__(self, ckpt, host, port, device, bgr_to_rgb=True):
        self.ckpt = ckpt
        self.host = host
        self.port = port
        self.device = torch.device(device)
        self.bgr_to_rgb = bgr_to_rgb

        self.policy = None

    def load_policy(self):
        print(f"Loading policy from: {self.ckpt}", flush=True)

        payload = torch.load(
            self.ckpt,
            map_location="cpu"
        )

        cfg = payload["cfg"]
        workspace_class = hydra.utils.get_class(cfg._target_)

        workspace = workspace_class(cfg)
        workspace.load_checkpoint(path=self.ckpt)

        self.policy = workspace.ema_model
        self.policy.eval()
        self.policy.to(self.device)

        print(f"Policy loaded on: {self.device}", flush=True)

    def preprocess(self, req):
        if "image" not in req:
            raise KeyError("Request does not contain 'image'")

        if "agent_pos" not in req:
            raise KeyError("Request does not contain 'agent_pos'")

        image_np = np.asarray(req["image"])
        agent_pos_np = np.asarray(
            req["agent_pos"],
            dtype=np.float32
        )

        if image_np.ndim != 4:
            raise ValueError(
                f"image must be [T,H,W,C], got {image_np.shape}"
            )

        if image_np.shape[-1] != 3:
            raise ValueError(
                f"image must have 3 channels, got {image_np.shape}"
            )

        if agent_pos_np.ndim != 2:
            raise ValueError(
                f"agent_pos must be [T,8], got {agent_pos_np.shape}"
            )

        if agent_pos_np.shape[1] != 8:
            raise ValueError(
                f"agent_pos dimension must be 8, "
                f"got {agent_pos_np.shape}"
            )

        if image_np.shape[0] != agent_pos_np.shape[0]:
            raise ValueError(
                "image and agent_pos must have the same "
                f"time dimension: {image_np.shape[0]} vs "
                f"{agent_pos_np.shape[0]}"
            )

        # Client 使用 cv_bridge 的 bgr8，因此在 Server 轉成 RGB。
        if self.bgr_to_rgb:
            image_np = image_np[..., ::-1].copy()

        image = torch.as_tensor(
            image_np,
            dtype=torch.float32,
            device=self.device
        ) / 255.0

        agent_pos = torch.as_tensor(
            agent_pos_np,
            dtype=torch.float32,
            device=self.device
        )

        # [T,H,W,C] -> [1,T,C,H,W]
        image = image.permute(0, 3, 1, 2).unsqueeze(0)

        # [T,8] -> [1,T,8]
        agent_pos = agent_pos.unsqueeze(0)

        return {
            "image": image,
            "agent_pos": agent_pos
        }

    def predict(self, req):
        obs = self.preprocess(req)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():
            result = self.policy.predict_action(obs)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_time = time.perf_counter() - start_time

        action = (
            result["action"][0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if action.ndim != 2 or action.shape[1] != 8:
            raise ValueError(
                f"Model returned invalid action shape: {action.shape}"
            )

        return {
            "action": action,
            "dt": inference_time,
            "action_shape": action.shape
        }

    def handle_client(self, conn, addr):
        print(f"Client connected: {addr}", flush=True)

        query_count = 0

        with conn:
            while True:
                try:
                    req = recv_msg(conn)

                    if req is None:
                        print(
                            f"Client disconnected: {addr}",
                            flush=True
                        )
                        break

                    result = self.predict(req)
                    send_msg(conn, result)

                    query_count += 1

                    print(
                        f"[{addr}] "
                        f"query={query_count}, "
                        f"action_shape={result['action_shape']}, "
                        f"dt={result['dt']:.3f}s",
                        flush=True
                    )

                except (ConnectionResetError, BrokenPipeError):
                    print(
                        f"Client connection lost: {addr}",
                        flush=True
                    )
                    break

                except Exception as error:
                    print(
                        f"Client error {addr}: {error}",
                        flush=True
                    )
                    break

    def serve_forever(self):
        self.load_policy()

        server = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )

        server.bind((self.host, self.port))
        server.listen(1)

        print(
            f"DP inference server listening on "
            f"{self.host}:{self.port}",
            flush=True
        )

        try:
            while True:
                conn, addr = server.accept()
                self.handle_client(conn, addr)

        except KeyboardInterrupt:
            print("Server stopped.", flush=True)

        finally:
            server.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default=DEFAULT_CKPT
    )

    # 現在 Server 與 Client 在同一台電腦，只開放本機連線。
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5005
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    parser.add_argument(
        "--no-bgr-to-rgb",
        action="store_true"
    )

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested, but CUDA is unavailable."
        )

    server = DPInferenceServer(
        ckpt=args.ckpt,
        host=args.host,
        port=args.port,
        device=args.device,
        bgr_to_rgb=not args.no_bgr_to_rgb
    )

    server.serve_forever()


if __name__ == "__main__":
    main()