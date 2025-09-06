#!/usr/bin/env python3
"""
遺伝的アルゴリズムの進化過程をGIFアニメーションとして作成するスクリプト
"""

import os
import re
from PIL import Image
import imageio
from pathlib import Path


def extract_generation_number(filename):
    """ファイル名から世代数を抽出"""
    match = re.search(r"best_score_[\d.]+_(\d+)\.png", filename)
    if match:
        return int(match.group(1))
    return 0


def create_evolution_gif(experiment_path, output_path, max_frames=20):
    """
    実験フォルダから進化のGIFを作成

    Args:
        experiment_path: 実験フォルダのパス
        output_path: 出力GIFファイルのパス
        max_frames: 最大フレーム数
    """
    best_dir = Path(experiment_path) / "best"

    if not best_dir.exists():
        print(f"Best directory not found: {best_dir}")
        return False

    # bestスコア画像を取得してソート
    image_files = []
    for file in best_dir.glob("best_score_*.png"):
        gen_num = extract_generation_number(file.name)
        image_files.append((gen_num, file))

    # 世代順にソート
    image_files.sort(key=lambda x: x[0])

    if len(image_files) == 0:
        print(f"No best score images found in {best_dir}")
        return False

    # フレーム数を制限
    if len(image_files) > max_frames:
        # 等間隔でサンプリング
        step = len(image_files) // max_frames
        image_files = image_files[::step][:max_frames]

    print(f"Creating GIF with {len(image_files)} frames from {experiment_path}")

    # 画像を読み込み
    frames = []
    for gen_num, file_path in image_files:
        try:
            img = Image.open(file_path)
            # 画像にテキストを追加（世代数）
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(img)
            text = f"Generation {gen_num}"

            # デフォルトフォントを使用
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
                )
            except:
                font = ImageFont.load_default()

            # テキストの位置を計算
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # 右上に配置
            x = img.width - text_width - 10
            y = 10

            # 背景を描画
            draw.rectangle(
                [x - 5, y - 2, x + text_width + 5, y + text_height + 2],
                fill=(0, 0, 0, 128),
            )
            draw.text((x, y), text, fill=(255, 255, 255), font=font)

            frames.append(img)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    if len(frames) == 0:
        print("No frames could be processed")
        return False

    # GIFを作成
    try:
        # PIL Imageをnumpy arrayに変換
        frame_arrays = []
        for frame in frames:
            frame_arrays.append(frame)

        # GIFとして保存
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=800,  # 0.8秒間隔
            loop=0,
        )

        print(f"GIF created successfully: {output_path}")
        return True

    except Exception as e:
        print(f"Error creating GIF: {e}")
        return False


def main():
    """メイン関数"""
    runs_dir = Path("runs")

    # 初音ミクの実験でGIFを作成
    experiments = [
        "hatsune_miku",
        "hatsune_miku_2",
        "hatsune_miku_3",
        "hatsune_miku_5_large_genes",
        "doraemon",
    ]

    for exp_name in experiments:
        exp_path = runs_dir / exp_name
        if exp_path.exists():
            output_file = f"{exp_name}_evolution.gif"
            print(f"\nProcessing {exp_name}...")
            success = create_evolution_gif(exp_path, output_file)
            if success:
                print(f"✓ Created {output_file}")
            else:
                print(f"✗ Failed to create {output_file}")
        else:
            print(f"Experiment not found: {exp_path}")


if __name__ == "__main__":
    main()

