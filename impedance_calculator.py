"""
シャントスルー法を中心としたインピーダンス計算モジュール
VNAで取得したSパラメータからDUTのインピーダンスを算出する
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ImpedanceData:
    """インピーダンス測定結果を保持するデータクラス"""
    frequencies: np.ndarray  # 周波数 [Hz]
    impedances: np.ndarray   # 複素インピーダンス [Ohm]
    magnitudes: np.ndarray   # インピーダンス絶対値 [Ohm]
    phases: np.ndarray       # インピーダンス位相 [deg]
    s11: np.ndarray          # S11 複素データ
    s21: np.ndarray          # S21 複素データ


class ImpedanceCalculator:
    """
    シャントスルー法・反射法・シリーズスルー法でインピーダンスを算出するクラス

    シャントスルー法ではDUTをVNAのポート同士に並列接続し、シリーズスルー法では
    直列接続してS21を測定する。
    """

    def __init__(self, z0: float = 50.0):
        """
        インピーダンス計算器の初期化

        Args:
            z0: 基準特性インピーダンス [Ohm] (既定値: 50Ω)
        """
        self.z0 = z0

    def calculate_from_s21_shunt(self, frequencies: np.ndarray, s21: np.ndarray) -> ImpedanceData:
        """
        シャントスルー法によるS21からのインピーダンス計算

        並列接続されたDUTはABCパラメータ

            A = 1, B = 0, C = 1/Z, D = 1

        に相当し、ABCD→S変換により

            S21 = 2 / (2 + Z0 / Z)

        が得られる。これをZについて解くと

            Z = (Z0 / 2) * (S21 / (1 - S21))

        となる。

        Args:
            frequencies: 周波数配列 [Hz]
            s21: 複素S21配列

        Returns:
            計算済みインピーダンスを含むImpedanceData
        """
        # Z = (Z0 / 2) * (S21 / (1 - S21)) に基づいて計算する
        # S21 ≈ 1 付近でのゼロ割りを回避する
        denominator = 1.0 - s21
        denominator_safe = np.where(np.abs(denominator) < 1e-10, 1e-10 + 0j, denominator)

        impedances = (self.z0 / 2.0) * (s21 / denominator_safe)

        # 絶対値と位相を算出する
        magnitudes = np.abs(impedances)
        phases = np.angle(impedances, deg=True)

        return ImpedanceData(
            frequencies=frequencies,
            impedances=impedances,
            magnitudes=magnitudes,
            phases=phases,
            s11=np.zeros_like(s21),  # 本計算では未使用
            s21=s21
        )

    def calculate_from_s11_reflection(self, frequencies: np.ndarray, s11: np.ndarray) -> ImpedanceData:
        """
        反射法によるS11からのインピーダンス計算

        反射測定では反射係数Γ = S11とZ0の関係から

            Z = Z0 * (1 + S11) / (1 - S11)

        が成立する。

        Args:
            frequencies: 周波数配列 [Hz]
            s11: 複素S11配列

        Returns:
            計算済みインピーダンスを含むImpedanceData
        """
        # Z = Z0 * (1 + S11) / (1 - S11) に基づいて計算する
        # 分母のゼロ割りを回避する
        denominator = 1.0 - s11
        denominator_safe = np.where(np.abs(denominator) < 1e-10, 1e-10 + 0j, denominator)

        impedances = self.z0 * (1.0 + s11) / denominator_safe

        # 絶対値と位相を算出する
        magnitudes = np.abs(impedances)
        phases = np.angle(impedances, deg=True)

        return ImpedanceData(
            frequencies=frequencies,
            impedances=impedances,
            magnitudes=magnitudes,
            phases=phases,
            s11=s11,
            s21=np.zeros_like(s11)  # 本計算では未使用
        )

    def calculate_from_s21_series(self, frequencies: np.ndarray, s21: np.ndarray) -> ImpedanceData:
        """
        シリーズスルー法によるS21からのインピーダンス計算

        直列接続されたDUTのABCD行列は

            [1  Z]
            [0  1]

        となり、ABCD→S変換より

            S21 = 2 / (2 + Z / Z0)

        が得られる。これをZについて解くと

            Z = 2 * Z0 * (1/S21 - 1)

        となる。

        Args:
            frequencies: 周波数配列 [Hz]
            s21: 複素S21配列

        Returns:
            計算済みインピーダンスを含むImpedanceData
        """
        # Z = 2*Z0*(1/S21 - 1) に基づいて計算する
        # S21 ≈ 0 付近でのゼロ割りを回避する
        s21_safe = np.where(np.abs(s21) < 1e-10, 1e-10 + 0j, s21)
        impedances = 2.0 * self.z0 * (1.0 / s21_safe - 1.0)

        # 絶対値と位相を算出する
        magnitudes = np.abs(impedances)
        phases = np.angle(impedances, deg=True)

        return ImpedanceData(
            frequencies=frequencies,
            impedances=impedances,
            magnitudes=magnitudes,
            phases=phases,
            s11=np.zeros_like(s21),
            s21=s21
        )

    def calculate_average(
        self,
        impedance_data_list: List[ImpedanceData],
        mode: str = "mean"
    ) -> ImpedanceData:
        """
        複数のインピーダンス測定結果を各種手法で平均化する

        測定点数が異なる場合は最小点数に合わせて切り詰める。

        Args:
            impedance_data_list: 平均対象のImpedanceDataリスト
            mode: 平均化モード
                - "mean": 単純平均 (既定値)
                - "median": 中央値 (外れ値に強い)
                - "trimmed": 10%両側トリム平均
                - "robust": MADに基づくロバスト平均

        Returns:
            平均後のImpedanceData
        """
        if not impedance_data_list:
            raise ValueError("Empty impedance data list")

        if len(impedance_data_list) == 1:
            # 1件のみの場合はそのまま返す
            return impedance_data_list[0]

        # 各測定の最小点数を求める
        lengths = [len(data.frequencies) for data in impedance_data_list]
        min_length = min(lengths)
        max_length = max(lengths)

        # 点数が異なる場合は警告を表示する
        if min_length != max_length:
            print(f"Warning: Measurements have different lengths ({min_length} to {max_length} points)")
            print(f"Trimming all measurements to {min_length} points for averaging")

        # 全てのデータを最小点数に揃える
        trimmed_data = []
        for data in impedance_data_list:
            trimmed = ImpedanceData(
                frequencies=data.frequencies[:min_length],
                impedances=data.impedances[:min_length],
                magnitudes=data.magnitudes[:min_length],
                phases=data.phases[:min_length],
                s11=data.s11[:min_length],
                s21=data.s21[:min_length]
            )
            trimmed_data.append(trimmed)

        # 基準となる周波数軸を最初の測定から取得する
        frequencies = trimmed_data[0].frequencies

        # 平均計算用に配列を構築する
        impedances_array = np.array([data.impedances for data in trimmed_data])
        s11_array = np.array([data.s11 for data in trimmed_data])
        s21_array = np.array([data.s21 for data in trimmed_data])

        # 平均化手法に応じて処理する
        if mode == "mean":
            avg_impedances = self._mean_average(impedances_array)
            avg_s11 = self._mean_average(s11_array)
            avg_s21 = self._mean_average(s21_array)
        elif mode == "median":
            avg_impedances = self._median_average(impedances_array)
            avg_s11 = self._median_average(s11_array)
            avg_s21 = self._median_average(s21_array)
        elif mode == "trimmed":
            avg_impedances = self._trimmed_mean_average(impedances_array)
            avg_s11 = self._trimmed_mean_average(s11_array)
            avg_s21 = self._trimmed_mean_average(s21_array)
        elif mode == "robust":
            avg_impedances = self._robust_average(impedances_array)
            avg_s11 = self._robust_average(s11_array)
            avg_s21 = self._robust_average(s21_array)
        else:
            raise ValueError(f"Unknown averaging mode: {mode}")

        # 平均化した複素インピーダンスから絶対値と位相を再計算する
        magnitudes = np.abs(avg_impedances)
        phases = np.angle(avg_impedances, deg=True)

        return ImpedanceData(
            frequencies=frequencies,
            impedances=avg_impedances,
            magnitudes=magnitudes,
            phases=phases,
            s11=avg_s11,
            s21=avg_s21
        )

    def _mean_average(self, data_array: np.ndarray) -> np.ndarray:
        """
        単純平均を計算する

        Args:
            data_array: 形状 (測定本数, 点数) の配列

        Returns:
            形状 (点数,) の平均化配列
        """
        return np.mean(data_array, axis=0)

    def _median_average(self, data_array: np.ndarray) -> np.ndarray:
        """
        中央値による平均 (外れ値に最も強い)

        複素数の場合は実部と虚部を別々に中央値処理する。

        Args:
            data_array: 形状 (測定本数, 点数) の配列

        Returns:
            形状 (点数,) の中央値配列
        """
        if np.iscomplexobj(data_array):
            # 複素データの場合は実部と虚部を別々に中央値化する
            real_median = np.median(data_array.real, axis=0)
            imag_median = np.median(data_array.imag, axis=0)
            return real_median + 1j * imag_median
        else:
            return np.median(data_array, axis=0)

    def _trimmed_mean_average(self, data_array: np.ndarray, trim_percent: float = 0.1) -> np.ndarray:
        """
        トリム平均 (外れ値を除去してから平均化する)

        平均化する前に上下trim_percent分の値を削除する。
        複素数の場合は絶対値に基づいてトリムを行う。

        Args:
            data_array: 形状 (測定本数, 点数) の配列
            trim_percent: 両端からトリムする割合 (既定値: 0.1 = 10%)

        Returns:
            形状 (点数,) のトリム平均配列
        """
        from scipy import stats

        if np.iscomplexobj(data_array):
            # 複素データの場合は絶対値に基づいてトリムする
            magnitudes = np.abs(data_array)

            # 各周波数点についてトリム平均を計算する
            result = np.zeros(data_array.shape[1], dtype=complex)

            for i in range(data_array.shape[1]):
                # 対象周波数点のデータを取得する
                point_data = data_array[:, i]
                point_mags = magnitudes[:, i]

                # 絶対値でソートする
                sorted_indices = np.argsort(point_mags)

                # トリムする件数を算出する
                n = len(point_data)
                trim_count = int(n * trim_percent)

                # 絶対値に基づいて両端をトリムする
                if trim_count > 0 and n > 2 * trim_count:
                    trimmed_indices = sorted_indices[trim_count:-trim_count]
                    result[i] = np.mean(point_data[trimmed_indices])
                else:
                    # トリムできない場合は単純平均を使う
                    result[i] = np.mean(point_data)

            return result
        else:
            # 実数データはscipyのトリム平均を利用する
            return stats.trim_mean(data_array, trim_percent, axis=0)

    def _robust_average(self, data_array: np.ndarray, mad_threshold: float = 3.0) -> np.ndarray:
        """
        MAD (Median Absolute Deviation) を用いたロバスト平均

        手順:
        1. 各周波数点で中央値を計算
        2. MAD (中央値からの絶対偏差の中央値) を計算
        3. mad_threshold * MAD を超える外れ値を除去
        4. 残りのデータを平均化

        複素数の場合は絶対値の偏差を基準として外れ値判定を行う。

        Args:
            data_array: 形状 (測定本数, 点数) の配列
            mad_threshold: 外れ値判定のしきい値 (既定値: 3.0)

        Returns:
            形状 (点数,) のロバスト平均配列
        """
        if np.iscomplexobj(data_array):
            # 複素データでは絶対値に基づいて外れ値を判定する
            magnitudes = np.abs(data_array)

            # 各周波数点でロバスト平均を計算する
            result = np.zeros(data_array.shape[1], dtype=complex)

            for i in range(data_array.shape[1]):
                # 対象周波数点のデータを取得する
                point_data = data_array[:, i]
                point_mags = magnitudes[:, i]

                # 絶対値の中央値とMADを算出する
                median_mag = np.median(point_mags)
                mad = np.median(np.abs(point_mags - median_mag))

                # ゼロ割りを避ける
                if mad < 1e-10:
                    # ほぼ同じ値なので単純平均を採用する
                    result[i] = np.mean(point_data)
                else:
                    # 外れ値ではないデータを抽出する
                    # 判定式: |絶対値 - 中央値| < mad_threshold * MAD
                    deviation = np.abs(point_mags - median_mag)
                    inlier_mask = deviation < (mad_threshold * mad)

                    # 外れ値を除去したデータで平均を取る
                    if np.sum(inlier_mask) > 0:
                        result[i] = np.mean(point_data[inlier_mask])
                    else:
                        # 全て外れ値になった場合は中央値を使用する
                        real_median = np.median(point_data.real)
                        imag_median = np.median(point_data.imag)
                        result[i] = real_median + 1j * imag_median

            return result
        else:
            # 実数データの処理
            result = np.zeros(data_array.shape[1])

            for i in range(data_array.shape[1]):
                point_data = data_array[:, i]

                # 中央値とMADを算出する
                median_val = np.median(point_data)
                mad = np.median(np.abs(point_data - median_val))

                if mad < 1e-10:
                    result[i] = np.mean(point_data)
                else:
                    # 外れ値ではないデータを抽出する
                    deviation = np.abs(point_data - median_val)
                    inlier_mask = deviation < (mad_threshold * mad)

                    if np.sum(inlier_mask) > 0:
                        result[i] = np.mean(point_data[inlier_mask])
                    else:
                        result[i] = median_val

            return result

    def calculate_with_reference(
        self,
        frequencies: np.ndarray,
        s21_dut: np.ndarray,
        s21_ref: np.ndarray
    ) -> ImpedanceData:
        """
        リファレンス測定 (スルー) を用いたインピーダンス計算

        スルー接続のリファレンス測定と比較することで、DUT測定を正規化する。

        Args:
            frequencies: 周波数配列 [Hz]
            s21_dut: DUTを接続した際のS21
            s21_ref: リファレンス (スルー) 測定のS21

        Returns:
            計算済みインピーダンスを含むImpedanceData
        """
        # リファレンスでS21を正規化する
        s21_ref_safe = np.where(np.abs(s21_ref) < 1e-10, 1e-10 + 0j, s21_ref)
        s21_normalized = s21_dut / s21_ref_safe

        # 正規化したS21からシャントスルー法でインピーダンスを算出する
        return self.calculate_from_s21_shunt(frequencies, s21_normalized)

    def calculate_parallel_components(
        self,
        frequencies: np.ndarray,
        impedances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        複素インピーダンスから並列RC定数を算出する

        並列RCモデル Z = R || (1/jωC) を仮定する。

        Args:
            frequencies: 周波数配列 [Hz]
            impedances: 複素インピーダンス配列

        Returns:
            (R_parallel, C_parallel) のタプル
        """
        # アドミタンス Y = 1/Z = 1/R + jωC を用いて解析する
        admittances = 1.0 / impedances

        conductances = np.real(admittances)  # G = 1/R
        susceptances = np.imag(admittances)  # B = ωC

        # R = 1/G
        r_parallel = np.where(np.abs(conductances) > 1e-10, 1.0 / conductances, np.inf)

        # C = B / ω
        omega = 2.0 * np.pi * frequencies
        c_parallel = susceptances / omega

        return r_parallel, c_parallel

    def calculate_series_components(
        self,
        frequencies: np.ndarray,
        impedances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        複素インピーダンスから直列RC定数を算出する

        直列RCモデル Z = R + 1/jωC を仮定する。

        Args:
            frequencies: 周波数配列 [Hz]
            impedances: 複素インピーダンス配列

        Returns:
            (R_series, C_series) のタプル
        """
        r_series = np.real(impedances)
        reactances = np.imag(impedances)

        # 容量リアクタンス X = -1/(ωC) より C = -1/(ωX)
        omega = 2.0 * np.pi * frequencies
        c_series = np.where(
            np.abs(reactances) > 1e-10,
            -1.0 / (omega * reactances),
            np.inf
        )

        return r_series, c_series


if __name__ == "__main__":
    # 動作確認用テストコード
    print("Impedance Calculator Test")
    print("=" * 70)

    z0 = 50.0
    calc = ImpedanceCalculator(z0=z0)
    frequencies = np.linspace(100e6, 200e6, 51)  # 100～200 MHz, 51点

    # ========== テスト1: シャント (S21) 法 ==========
    print("\n[Test 1] Shunt (S21) method")
    print("-" * 70)

    z_test_shunt = 100.0  # 100 Ω
    # シャント法の式: S21 = 2*Z / (2*Z + Z0)
    s21_test_shunt = 2 * z_test_shunt / (2 * z_test_shunt + z0)
    s21_array_shunt = np.full(len(frequencies), s21_test_shunt, dtype=complex)

    print(f"Input impedance:     {z_test_shunt} Ohm")
    print(f"Calculated S21:      {s21_test_shunt:.6f}")
    expected_s21_shunt = 2 * z_test_shunt / (2 * z_test_shunt + z0)
    print(f"Expected S21:        {expected_s21_shunt:.6f}")

    impedance_data_shunt = calc.calculate_from_s21_shunt(frequencies, s21_array_shunt)

    print(f"\nReconstructed impedance: {impedance_data_shunt.magnitudes[0]:.2f} Ohm")
    print(f"Error:                   {abs(impedance_data_shunt.magnitudes[0] - z_test_shunt):.6f} Ohm")
    assert abs(impedance_data_shunt.magnitudes[0] - z_test_shunt) < 0.01, "Shunt calculation error!"
    print("[OK] Shunt method verified")

    # ========== テスト2: 反射 (S11) 法 ==========
    print("\n[Test 2] Reflection (S11) method")
    print("-" * 70)

    z_test_s11 = 75.0  # 75 Ω
    # 反射法の式: S11 = (Z - Z0) / (Z + Z0)
    s11_test = (z_test_s11 - z0) / (z_test_s11 + z0)
    s11_array = np.full(len(frequencies), s11_test, dtype=complex)

    print(f"Input impedance:     {z_test_s11} Ohm")
    print(f"Calculated S11:      {s11_test:.6f}")
    print(f"Expected S11:        {(75-50)/(75+50):.6f}")

    impedance_data_s11 = calc.calculate_from_s11_reflection(frequencies, s11_array)

    print(f"\nReconstructed impedance: {impedance_data_s11.magnitudes[0]:.2f} Ohm")
    print(f"Error:                   {abs(impedance_data_s11.magnitudes[0] - z_test_s11):.6f} Ohm")
    assert abs(impedance_data_s11.magnitudes[0] - z_test_s11) < 0.01, "S11 calculation error!"
    print("[OK] S11 method verified")

    # ========== テスト3: シリーズ (S21) 法 ==========
    print("\n[Test 3] Series (S21) method")
    print("-" * 70)

    z_test_series = 50.0  # 50 Ω
    # シリーズ法の式: S21 = 2*Z0 / (2*Z0 + Z)
    s21_test_series = 2 * z0 / (2 * z0 + z_test_series)
    s21_array_series = np.full(len(frequencies), s21_test_series, dtype=complex)

    print(f"Input impedance:     {z_test_series} Ohm")
    print(f"Calculated S21:      {s21_test_series:.6f}")
    expected_s21_series = 2 * z0 / (2 * z0 + z_test_series)
    print(f"Expected S21:        {expected_s21_series:.6f}")

    impedance_data_series = calc.calculate_from_s21_series(frequencies, s21_array_series)

    print(f"\nReconstructed impedance: {impedance_data_series.magnitudes[0]:.2f} Ohm")
    print(f"Error:                   {abs(impedance_data_series.magnitudes[0] - z_test_series):.6f} Ohm")
    assert abs(impedance_data_series.magnitudes[0] - z_test_series) < 0.01, "Series calculation error!"
    print("[OK] Series method verified")

    # ========== テスト4: 整合性チェック ==========
    print("\n[Test 4] Same DUT measured with S11 and Shunt should give similar results")
    print("-" * 70)

    z_dut = 100.0  # 100 ΩのDUT

    # S11測定
    s11_dut = (z_dut - z0) / (z_dut + z0)
    s11_array_dut = np.full(len(frequencies), s11_dut, dtype=complex)
    z_from_s11 = calc.calculate_from_s11_reflection(frequencies, s11_array_dut)

    # シャント測定
    s21_dut = 2 * z_dut / (2 * z_dut + z0)
    s21_array_dut = np.full(len(frequencies), s21_dut, dtype=complex)
    z_from_shunt = calc.calculate_from_s21_shunt(frequencies, s21_array_dut)

    print(f"DUT impedance:           {z_dut} Ohm")
    print(f"From S11 (Reflection):   {z_from_s11.magnitudes[0]:.2f} Ohm")
    print(f"From S21 (Shunt):        {z_from_shunt.magnitudes[0]:.2f} Ohm")
    print(f"Difference:              {abs(z_from_s11.magnitudes[0] - z_from_shunt.magnitudes[0]):.6f} Ohm")
    assert abs(z_from_s11.magnitudes[0] - z_from_shunt.magnitudes[0]) < 0.01, "S11 and Shunt should give same result!"
    print("[OK] Consistency verified - S11 and Shunt give identical results")

    print("\n" + "=" * 70)
    print("All tests passed! [OK]")
    print("=" * 70)
