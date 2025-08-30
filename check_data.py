#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
データセット整合性チェックスクリプト
確認項目:
1) ルート直下に "FovL", "FovS_180", "FovS_360" が存在し、それぞれ subject_0000 ～ subject_0049 を含むこと
2) 各 subject_00xx ディレクトリ配下に phase_00 ～ phase_04 および prior が存在すること
3) prior フォルダに 384 個の *.img が存在すること
4) 各 phase ディレクトリに gt と img が存在すること
5) 各 gt と img にそれぞれ 384 個の *.img が存在すること
"""

import os
import glob
from config import DATASET_CONFIG


# 期待される設定
EXPECTED_FOVS = ["FovL", "FovS_180", "FovS_360"]
EXPECTED_SUBJECTS = [f"subject_{i:04d}" for i in range(50)]  # 0000 - 0049
EXPECTED_PHASES = [f"phase_{i:02d}" for i in range(5)]       # phase_00 - phase_04
EXPECTED_IMG_COUNT = 384

def check_dataset(root_data):
    problems = 0

    # 1) FOV ディレクトリの存在確認
    for fov in EXPECTED_FOVS:
        fov_path = os.path.join(root_data, fov)
        if not os.path.isdir(fov_path):
            print(f"[ERROR] FOV ディレクトリ未検出: {fov_path}")
            problems += 1
            continue

        # 現在存在する被験者ディレクトリを取得
        found_subjects = sorted([d for d in os.listdir(fov_path) if os.path.isdir(os.path.join(fov_path, d))])
        # 期待される被験者との差分を確認
        missing = [s for s in EXPECTED_SUBJECTS if s not in found_subjects]
        extra = [s for s in found_subjects if s not in EXPECTED_SUBJECTS]

        if missing:
            print(f"[ERROR] {fov}: 欠落被験者 ({len(missing)}): {', '.join(missing)}")
            problems += 1
        if extra:
            # 余分なディレクトリがあれば報告（フォーマット違いや不要データ）
            print(f"[WARN] {fov}: 予期しない被験者ディレクトリ ({len(extra)}): {', '.join(extra)}")

        # 逐一被験者ごとにチェック
        for subject in EXPECTED_SUBJECTS:
            subject_path = os.path.join(fov_path, subject)
            if not os.path.isdir(subject_path):
                # 既に missing リストで報告済みだが念のため個別出力はせず continue
                continue

            # 2) phase と prior の存在確認
            # prior
            prior_dir = os.path.join(subject_path, "prior")
            if not os.path.isdir(prior_dir):
                print(f"[ERROR] {fov}/{subject}: prior ディレクトリが存在しません: {prior_dir}")
                problems += 1
            else:
                # 3) prior のファイル数確認
                prior_files = sorted(glob.glob(os.path.join(prior_dir, "*.img")))
                prior_count = len(prior_files)
                if prior_count != EXPECTED_IMG_COUNT:
                    print(f"[ERROR] {fov}/{subject}/prior: *.img ファイル数不一致 (期待: {EXPECTED_IMG_COUNT}, 実際: {prior_count})")
                    problems += 1

            # phases
            for phase in EXPECTED_PHASES:
                phase_path = os.path.join(subject_path, phase)
                if not os.path.isdir(phase_path):
                    print(f"[ERROR] {fov}/{subject}: フェーズディレクトリ欠落: {phase_path}")
                    problems += 1
                    # このフェーズは存在しないため次へ
                    continue

                # 4) phase 内に img と gt が存在するか
                img_dir = os.path.join(phase_path, "img")
                gt_dir = os.path.join(phase_path, "gt")
                if not os.path.isdir(img_dir):
                    print(f"[ERROR] {fov}/{subject}/{phase}: img ディレクトリが存在しません: {img_dir}")
                    problems += 1
                if not os.path.isdir(gt_dir):
                    print(f"[ERROR] {fov}/{subject}/{phase}: gt ディレクトリが存在しません: {gt_dir}")
                    problems += 1

                # 5) img と gt のファイル数確認（存在する場合のみ）
                if os.path.isdir(img_dir) and os.path.isdir(gt_dir):
                    img_files = sorted(glob.glob(os.path.join(img_dir, "*.img")))
                    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.img")))
                    img_count = len(img_files)
                    gt_count = len(gt_files)

                    if img_count != EXPECTED_IMG_COUNT:
                        print(f"[ERROR] {fov}/{subject}/{phase}/img: *.img ファイル数不一致 (期待: {EXPECTED_IMG_COUNT}, 実際: {img_count})")
                        problems += 1
                    if gt_count != EXPECTED_IMG_COUNT:
                        print(f"[ERROR] {fov}/{subject}/{phase}/gt: *.img ファイル数不一致 (期待: {EXPECTED_IMG_COUNT}, 実際: {gt_count})")
                        problems += 1

                    # 追加チェック: img と gt の数が一致しているか（元コードで必要としているチェック）
                    if img_count != gt_count:
                        print(f"[ERROR] {fov}/{subject}/{phase}: img と gt のファイル数が一致しません (img: {img_count}, gt: {gt_count})")
                        problems += 1

    # 最終サマリ
    if problems == 0:
        print("[OK] チェック完了: 問題は検出されませんでした。")
    else:
        print(f"[SUMMARY] チェック完了: 問題数 = {problems}")

if __name__ == "__main__":
    root = DATASET_CONFIG['data_root']
    check_dataset(root)
