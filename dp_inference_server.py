#!/usr/bin/env python3

import argparse
import socket
import pickle
import struct
import time

import hydra
import numpy as np
import torch


DEFAULT_CKPT = (
    "/home/itri2026-3090/diffusion_policy/data/outputs/2026.07.08/future6_train_diffusion_unet_image_so101_image/checkpoints/best.ckpt"
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
        print(f"Loading policy from: {self.ckpt}")
        payload = torch.load(self.ckpt, map_location="cpu")

        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)

        workspace = cls(cfg)
        workspace.load_checkpoint(path=self.ckpt)

        self.policy = workspace.ema_model
        self.policy.eval()
        self.policy.to(self.device)

        print("Policy loaded.")

    def preprocess(self, req):
        image_np = np.asarray(req["image"])
        agent_pos_np = np.asarray(req["agent_pos"], dtype=np.float32)

        if image_np.ndim != 4:
            raise ValueError(f"image must be [T,H,W,C], got {image_np.shape}")

        if agent_pos_np.ndim != 2 or agent_pos_np.shape[1] != 8:
            raise ValueError(
                f"agent_pos must be [T,8], got {agent_pos_np.shape}"
            )

        if self.bgr_to_rgb:
            image_np = image_np[..., ::-1].copy()

        image = torch.as_tensor(
            image_np,
            dtype=torch.float32,
            device=self.device,
        ) / 255.0

        agent_pos = torch.as_tensor(
            agent_pos_np,
            dtype=torch.float32,
            device=self.device,
        )

        image = image.permute(0, 3, 1, 2).unsqueeze(0)
        agent_pos = agent_pos.unsqueeze(0)

        return {
            "image": image,
            "agent_pos": agent_pos,
        }

    def predict(self, req):
        obs = self.preprocess(req)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()

        with torch.inference_mode():
            result = self.policy.predict_action(obs)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        dt = time.time() - t0

        action = result["action"][0].detach().cpu().numpy()

        return {
            "action": action,
            "dt": dt,
            "action_shape": action.shape,
        }

    def handle_client(self, conn, addr):
        print("Client connected:", addr)
        count = 0

        with conn:
            while True:
                try:
                    req = recv_msg(conn)

                    if req is None:
                        print("Client disconnected:", addr)
                        break

                    result = self.predict(req)
                    send_msg(conn, result)

                    count += 1

                    if count % 10 == 0:
                        print(
                            f"[{addr}] query={count}, "
                            f"action_shape={result['action_shape']}, "
                            f"dt={result['dt']:.3f}s"
                        )

                except Exception as e:
                    print(f"Client error {addr}: {e}")
                    break

    def serve_forever(self):
        self.load_policy()

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)

        print(f"DP inference server listening on {self.host}:{self.port}")

        try:
            while True:
                conn, addr = server.accept()
                self.handle_client(conn, addr)

        except KeyboardInterrupt:
            print("Server stopped.")

        finally:
            server.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-bgr-to-rgb", action="store_true")

    args = parser.parse_args()

    server = DPInferenceServer(
        ckpt=args.ckpt,
        host=args.host,
        port=args.port,
        device=args.device,
        bgr_to_rgb=not args.no_bgr_to_rgb,
    )

    server.serve_forever()


if __name__ == "__main__":
    main()