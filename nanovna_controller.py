"""
NanoVNA-F v2 シリアル通信コントローラ
NanoVNA-F v2 とのシリアル通信を管理するモジュール
"""

import serial
import serial.tools.list_ports
import time
import re
from typing import Optional, List, Tuple
import numpy as np


class NanoVNAController:
    """NanoVNA-F v2 デバイスとの通信を扱うコントローラクラス"""

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, timeout: float = 5.0, debug: bool = False):
        """
        NanoVNA コントローラを初期化する

        Args:
            port: COMポート名 (例: 'COM3')。None の場合は自動検出
            baudrate: ボーレート (既定: 115200)
            timeout: シリアル通信のタイムアウト [秒]
            debug: デバッグ出力を有効化するかどうか
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.debug = debug
        self.serial: Optional[serial.Serial] = None

    def auto_detect_port(self) -> Optional[str]:
        """
        Windows 上で NanoVNA の COM ポートを自動検出する

        Returns:
            見つかった場合はポート名、見つからない場合は None
        """
        ports = serial.tools.list_ports.comports()

        # VID/PID またはデバイス説明から NanoVNA を探す
        for port in ports:
            # NanoVNA-F v2 は USB Serial Device として認識されることが多い
            if self.debug:
                print(f"Found port: {port.device} - {port.description} - VID:PID={port.vid}:{port.pid}")

            # NanoVNA-F v2 の一般的な VID:PID は 0483:5740 (STM32 Virtual COM Port)
            if port.vid == 0x0483 and port.pid == 0x5740:
                if self.debug:
                    print(f"NanoVNA-F v2 detected on {port.device}")
                return port.device

            # 判別できなければ説明文を確認する
            if "NanoVNA" in port.description or "STM32 Virtual ComPort" in port.description:
                if self.debug:
                    print(f"Possible NanoVNA device on {port.device}")
                return port.device

        # 見つからない場合は利用可能な最初の COM ポートを返す
        if ports:
            if self.debug:
                print(f"No NanoVNA detected, using first available port: {ports[0].device}")
            return ports[0].device

        return None

    def connect(self) -> bool:
        """
        NanoVNA デバイスへ接続する

        Returns:
            接続に成功した場合は True、失敗した場合は False
        """
        try:
            if self.port is None:
                self.port = self.auto_detect_port()

            if self.port is None:
                print("Error: No COM port available")
                return False

            if self.debug:
                print(f"Connecting to {self.port}...")

            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )

            # デバイスが応答可能になるまで待つ
            time.sleep(0.5)

            # バッファをクリア
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()

            if self.debug:
                print(f"Connected to {self.port}")

            return True

        except Exception as e:
            print(f"Error connecting to {self.port}: {e}")
            return False

    def disconnect(self):
        """NanoVNA デバイスとの接続を切断する"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            if self.debug:
                print("Disconnected from NanoVNA")

    def send_command(self, command: str) -> str:
        """
        NanoVNA にコマンドを送り、応答を取得する

        Args:
            command: 改行無しのコマンド文字列

        Returns:
            応答文字列
        """
        if not self.serial or not self.serial.is_open:
            raise RuntimeError("Not connected to NanoVNA")

        # バッファをクリア
        self.serial.reset_input_buffer()

        # コマンドに改行を付けて送信
        cmd_bytes = (command + '\n').encode('ascii')
        self.serial.write(cmd_bytes)

        if self.debug:
            print(f"TX: {command}")

        # 応答を読み取る
        response_lines = []
        start_time = time.time()

        while True:
            if time.time() - start_time > self.timeout:
                break

            if self.serial.in_waiting > 0:
                line = self.serial.readline().decode('ascii', errors='ignore').strip()
                if line:
                    response_lines.append(line)
                    if self.debug:
                        print(f"RX: {line}")

                    # 'ch>' プロンプトで応答終了を判定
                    if line.endswith('ch>'):
                        break
            else:
                time.sleep(0.01)

        return '\n'.join(response_lines)

    def get_version(self) -> str:
        """NanoVNA のファームウェアバージョンを取得する"""
        return self.send_command("version")

    def set_sweep_parameters(self, start_freq: int, stop_freq: int, points: int):
        """
        掃引パラメータを設定する

        Args:
            start_freq: 開始周波数 [Hz]
            stop_freq: 終了周波数 [Hz]
            points: 測定ポイント数 (11-301)
        """
        if points < 11 or points > 301:
            raise ValueError("Points must be between 11 and 301")

        command = f"sweep {start_freq} {stop_freq} {points}"
        response = self.send_command(command)
        return response

    def scan(self, start_freq: int, stop_freq: int, points: int, outmask: int = 7) -> List[Tuple[float, complex, complex]]:
        """
        掃引を実行して測定データを取得する

        Args:
            start_freq: 開始周波数 [Hz]
            stop_freq: 終了周波数 [Hz]
            points: 測定ポイント数 (11-301)
            outmask: 出力フォーマットマスク
                0: 出力なし
                1: 周波数のみ
                2: S11 データのみ
                3: 周波数 + S11 データ
                4: S21 データのみ
                5: 周波数 + S21 データ
                6: S11 データ + S21 データ
                7: 周波数 + S11 データ + S21 データ (推奨)

        Returns:
            (周波数, S11複素数, S21複素数) のタプル一覧
        """
        if points < 11 or points > 301:
            raise ValueError("Points must be between 11 and 301")

        command = f"scan {start_freq} {stop_freq} {points} {outmask}"

        if self.debug:
            print(f"Scanning: {start_freq/1e6:.1f} MHz to {stop_freq/1e6:.1f} MHz, {points} points")

        # ポイント数に応じて動的タイムアウトを設定 (概ね 1ポイント 0.1 秒 + バッファ)
        scan_timeout = max(self.timeout, points * 0.15 + 5)

        # バッファをクリア
        self.serial.reset_input_buffer()

        # コマンドを送信
        cmd_bytes = (command + '\n').encode('ascii')
        self.serial.write(cmd_bytes)

        if self.debug:
            print(f"TX: {command}")
            print(f"Timeout set to {scan_timeout:.1f}s for {points} points")

        # 応答を読み取る
        data = []
        start_time = time.time()
        line_count = 0
        last_data_time = time.time()
        idle_timeout = 3.0  # 3 秒間データが来なければタイムアウト

        while True:
            elapsed = time.time() - start_time
            idle_time = time.time() - last_data_time

            # 総タイムアウトを確認
            if elapsed > scan_timeout:
                if self.debug:
                    print(f"Total timeout reached: received {line_count}/{points} lines in {elapsed:.1f}s")
                break

            # 無通信タイムアウトを確認
            if idle_time > idle_timeout and line_count > 0:
                if self.debug:
                    print(f"Idle timeout: no data for {idle_time:.1f}s, received {line_count}/{points} lines")
                break

            if self.serial.in_waiting > 0:
                line = self.serial.readline().decode('ascii', errors='ignore').strip()

                if not line:
                    continue

                last_data_time = time.time()  # 無通信タイマーをリセット

                if self.debug:
                    print(f"RX: {line}")

                # コマンドのエコーをスキップ
                if line.startswith('scan'):
                    continue

                # プロンプトが来たら終了
                if 'ch>' in line:
                    if self.debug:
                        print(f"Prompt detected, scan complete")
                    break

                # データ行を解析
                # フォーマット: <frequency> <s11_real> <s11_imag> <s21_real> <s21_imag>
                try:
                    parts = line.split()
                    if len(parts) >= 5:
                        freq = float(parts[0])
                        s11_real = float(parts[1])
                        s11_imag = float(parts[2])
                        s21_real = float(parts[3])
                        s21_imag = float(parts[4])

                        s11 = complex(s11_real, s11_imag)
                        s21 = complex(s21_real, s21_imag)

                        data.append((freq, s11, s21))
                        line_count += 1

                        # 期待したポイント数に達したか確認
                        if line_count >= points:
                            if self.debug:
                                print(f"Received all {points} expected points")
                            break
                except (ValueError, IndexError) as e:
                    if self.debug:
                        print(f"Error parsing line: {line} - {e}")
                    continue
            else:
                time.sleep(0.01)

        if self.debug:
            print(f"Scan completed: received {len(data)}/{points} data points in {time.time() - start_time:.2f}s")

        # ポイント数が不足している場合は警告
        if len(data) < points:
            print(f"Warning: Expected {points} points but received {len(data)} points")

        return data

    def scan_logarithmic(self, start_freq: int, stop_freq: int, points: int, outmask: int = 7) -> List[Tuple[float, complex, complex]]:
        """
        対数間隔の掃引を行う

        NanoVNA-F v2 のコマンドは線形掃引のみ対応のため、本メソッドでは
        対数間隔で分割した複数の帯域を線形掃引し、結果を結合する。

        Args:
            start_freq: 開始周波数 [Hz]
            stop_freq: 終了周波数 [Hz]
            points: 総測定ポイント数 (11-301)
            outmask: 出力フォーマットマスク (scan メソッドと同じ)

        Returns:
            (周波数, S11複素数, S21複素数) のタプル一覧
        """
        if points < 11 or points > 301:
            raise ValueError("Points must be between 11 and 301")

        if self.debug:
            print(f"Logarithmic scan: {start_freq/1e6:.1f} MHz to {stop_freq/1e6:.1f} MHz, {points} points")

        # 対数間隔の周波数リストを生成
        log_start = np.log10(start_freq)
        log_stop = np.log10(stop_freq)
        log_freqs = np.logspace(log_start, log_stop, points)

        # 周波数帯域を複数に分割して掃引する
        # 対数分布を保ちつつ帯域数を抑えて性能を確保
        num_bands = min(3, max(3, points // 4))  # ポイント数に応じて 3～10 帯域

        band_edges = np.logspace(log_start, log_stop, num_bands + 1)

        if self.debug:
            print(f"Using {num_bands} frequency bands for logarithmic scan")

        all_data = []

        for i in range(num_bands):
            band_start = int(band_edges[i])
            band_stop = int(band_edges[i + 1])

            # 対数密度に比例するように帯域のポイント数を算出
            # 目標周波数が帯域内にいくつ含まれるかをカウント
            band_points = np.sum((log_freqs >= band_start) & (log_freqs <= band_stop))
            band_points = max(11, min(band_points + 5, 101))  # 各帯域で最低 11、最大 101 ポイント

            if self.debug:
                print(f"Band {i+1}/{num_bands}: {band_start/1e6:.3f}-{band_stop/1e6:.3f} MHz, {band_points} points")

            # 各帯域を掃引
            try:
                band_data = self.scan(band_start, band_stop, band_points, outmask)
                all_data.extend(band_data)
            except Exception as e:
                if self.debug:
                    print(f"Error scanning band {i+1}: {e}")
                continue

            # 帯域間で少し待機
            time.sleep(0.05)

        if not all_data:
            if self.debug:
                print("No data collected from logarithmic scan")
            return []

        # 周波数でソート
        all_data.sort(key=lambda x: x[0])

        # 帯域境界の重複を除去 (先に取得した方を残す)
        unique_data = []
        last_freq = -1
        freq_tolerance = 0.01  # 1% を重複判定のしきい値とする

        for freq, s11, s21 in all_data:
            if last_freq < 0 or abs(freq - last_freq) / last_freq > freq_tolerance:
                unique_data.append((freq, s11, s21))
                last_freq = freq

        if self.debug:
            print(f"Logarithmic scan collected {len(all_data)} raw points, {len(unique_data)} unique points")

        # 必要に応じて要求された対数周波数に補間
        # 要求ポイント数に合わせるための処理
        if len(unique_data) > points:
            # 要求されたポイント数に間引く
            indices = np.linspace(0, len(unique_data) - 1, points, dtype=int)
            final_data = [unique_data[i] for i in indices]
        else:
            # 収集したデータをそのまま利用
            final_data = unique_data

        if self.debug:
            print(f"Final logarithmic scan: {len(final_data)} points")

        return final_data

    def scan_multi_band(
        self,
        bands: List[Tuple[int, int, int]],
        outmask: int = 7,
        sweep_mode: str = "linear",
        calibration_ids: Optional[List[Optional[int]]] = None
    ) -> List[Tuple[float, complex, complex]]:
        """
        複数の周波数帯を個別設定で掃引する

        帯域ごとに測定ポイントや掃引モードを変えられるため、広帯域を詳細に観測するときに有効。

        Args:
            bands: 各帯域の (開始周波数, 終了周波数, ポイント数) のリスト
                   例: [(100e3, 1e6, 100), (1e6, 10e6, 100), ...]
            outmask: 出力フォーマットマスク (scan メソッドと同じ)
            sweep_mode: 各帯域の掃引モード。"linear" または "logarithmic"
            calibration_ids: 帯域ごとのキャリブレーションスロットID (0-6) のリスト。
                              None または空リストの場合はキャリブレーション未適用。
                              例: [0, 1, 2, 3] で各帯域に異なるキャリブレーションを適用

        Returns:
            (周波数, S11複素数, S21複素数) の結合リスト
        """
        if not bands:
            raise ValueError("At least one band must be specified")

        # キャリブレーションIDの整合性チェック
        if calibration_ids:
            if len(calibration_ids) != len(bands):
                raise ValueError(f"Number of calibration IDs ({len(calibration_ids)}) must match number of bands ({len(bands)})")

        if self.debug:
            print(f"Multi-band scan: {len(bands)} bands, {sweep_mode} sweep per band")
            total_points = sum(b[2] for b in bands)
            print(f"Total points: {total_points}")
            if calibration_ids:
                print(f"Calibration IDs: {calibration_ids}")

        all_data = []
        scan_function = self.scan_logarithmic if sweep_mode == "logarithmic" else self.scan

        for i, (start_freq, stop_freq, points) in enumerate(bands):
            if self.debug:
                print(f"\nBand {i+1}/{len(bands)}: {start_freq/1e6:.3f} - {stop_freq/1e6:.3f} MHz, {points} points")

            # 指定されていれば帯域ごとのキャリブレーションを適用
            if calibration_ids and i < len(calibration_ids) and calibration_ids[i] is not None:
                cal_id = calibration_ids[i]
                if self.debug:
                    print(f"  Applying calibration ID: {cal_id}")
                try:
                    self.recall_calibration(cal_id)
                except Exception as e:
                    print(f"Warning: Failed to recall calibration {cal_id}: {e}")
                    if self.debug:
                        import traceback
                        traceback.print_exc()

            try:
                # 帯域を掃引
                band_data = scan_function(int(start_freq), int(stop_freq), points, outmask)

                if band_data:
                    all_data.extend(band_data)
                    if self.debug:
                        print(f"  Collected {len(band_data)} points from band {i+1}")
                else:
                    print(f"Warning: No data collected from band {i+1}")

            except Exception as e:
                print(f"Error scanning band {i+1}: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                continue

            # 帯域間で短時間待機
            if i < len(bands) - 1:
                time.sleep(0.1)

        if not all_data:
            if self.debug:
                print("No data collected from multi-band scan")
            return []

        # 周波数でソートして順序を揃える
        all_data.sort(key=lambda x: x[0])

        # 帯域境界で重複したデータを除去 (先着を優先)
        unique_data = []
        last_freq = -1
        freq_tolerance = 0.001  # 0.1% を重複判定のしきい値とする

        for freq, s11, s21 in all_data:
            if last_freq < 0 or abs(freq - last_freq) / max(last_freq, freq) > freq_tolerance:
                unique_data.append((freq, s11, s21))
                last_freq = freq

        if self.debug:
            print(f"\nMulti-band scan completed:")
            print(f"  Total raw points: {len(all_data)}")
            print(f"  Unique points: {len(unique_data)}")
            print(f"  Frequency range: {unique_data[0][0]/1e6:.3f} - {unique_data[-1][0]/1e6:.3f} MHz")

        return unique_data

    def get_frequencies(self) -> List[float]:
        """
        現在設定されている掃引周波数リストを取得する

        Returns:
            周波数 [Hz] のリスト
        """
        response = self.send_command("frequencies")

        frequencies = []
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.endswith('ch>') and not line.startswith('frequencies'):
                try:
                    freq = float(line)
                    frequencies.append(freq)
                except ValueError:
                    continue

        return frequencies

    def calibration_load(self):
        """キャリブレーション LOAD を開始"""
        return self.send_command("cal load")

    def calibration_open(self):
        """キャリブレーション OPEN を開始"""
        return self.send_command("cal open")

    def calibration_short(self):
        """キャリブレーション SHORT を開始"""
        return self.send_command("cal short")

    def calibration_thru(self):
        """キャリブレーション THRU を開始"""
        return self.send_command("cal thru")

    def calibration_done(self):
        """キャリブレーションを完了"""
        return self.send_command("cal done")

    def calibration_on(self):
        """キャリブレーションを有効化"""
        return self.send_command("cal on")

    def calibration_off(self):
        """キャリブレーションを無効化"""
        return self.send_command("cal off")

    def save_calibration(self, slot_id: int):
        """
        キャリブレーションデータをスロットへ保存する

        Args:
            slot_id: キャリブレーションスロット番号 (0-6)
        """
        if slot_id < 0 or slot_id > 6:
            raise ValueError("Calibration slot must be between 0 and 6")

        command = f"save {slot_id}"
        response = self.send_command(command)

        if self.debug:
            print(f"Saved calibration to slot {slot_id}")

        return response

    def recall_calibration(self, slot_id: int):
        """
        スロットからキャリブレーションデータを呼び出す

        Args:
            slot_id: キャリブレーションスロット番号 (0-6)
        """
        if slot_id < 0 or slot_id > 6:
            raise ValueError("Calibration slot must be between 0 and 6")

        command = f"recall {slot_id}"
        response = self.send_command(command)

        if self.debug:
            print(f"Recalled calibration from slot {slot_id}")

        # キャリブレーションが反映されるよう少し待つ
        time.sleep(0.2)

        return response

    def __enter__(self):
        """コンテキストマネージャ: enter"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャ: exit"""
        self.disconnect()


if __name__ == "__main__":
    # 動作確認用テストコード
    print("NanoVNA-F v2 Controller Test")
    print("-" * 50)

    with NanoVNAController(debug=True) as vna:
        print("\nGetting version...")
        version = vna.get_version()
        print(f"Version: {version}")

        print("\nPerforming test scan...")
        start = 100_000_000  # 100 MHz
        stop = 200_000_000   # 200 MHz
        points = 51

        data = vna.scan(start, stop, points, outmask=7)

        print(f"\nReceived {len(data)} data points")
        if data:
            print("\nFirst 5 data points:")
            for i, (freq, s11, s21) in enumerate(data[:5]):
                print(f"  {i+1}. Freq: {freq/1e6:.2f} MHz, S11: {s11:.6f}, S21: {s21:.6f}")
