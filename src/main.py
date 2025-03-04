import tkinter as tk
from tkinter import ttk
import os
import sys
import joblib
import json
from datetime import datetime

# モジュールのインポートパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 現在のファイルのディレクトリパスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# プロジェクトのルートディレクトリをPythonのパスに追加
sys.path.append(current_dir)

project_root = "C:\\banei-ai-app"
sys.path.append(project_root)

# 確認のため表示
print(f"パスを追加しました: {project_root}")
print(f"Pythonの検索パス: {sys.path}")

class BaneiAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ばんえい競馬AI予想システム")
        self.root.geometry("900x700")
        
        # モデルのロード
        self.model_path = "src/models/optimized_lightgbm_model.pkl"

        self.load_model()
        
        # GUIコンポーネントの初期化
        self.setup_gui()
        
        # データの初期化
        self.horse_data = {}
        self.prediction_results = {}
    
    def load_model(self):
        """予測モデルをロードする"""
        try:
            self.model = joblib.load(self.model_path)
            print(f"モデルを正常にロードしました: {self.model_path}")
        except Exception as e:
            print(f"モデルのロード中にエラーが発生しました: {str(e)}")
            # エラーがあってもプログラムを動かせるようにNoneを設定
            self.model = None
    
    def setup_gui(self):
        """GUIコンポーネントをセットアップする"""
        # ナビゲーションタブ
        self.tab_control = ttk.Notebook(self.root)
        
        # 予測タブ
        self.prediction_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.prediction_tab, text="レース予想")
        self.setup_prediction_tab()
        
        # 履歴タブ
        self.history_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.history_tab, text="予測履歴")
        self.setup_history_tab()
        
        # 設定タブ
        self.settings_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.settings_tab, text="設定")
        self.setup_settings_tab()
        
        self.tab_control.pack(expand=1, fill="both")
    
    def setup_prediction_tab(self):
        """予測タブのコンポーネントをセットアップする"""
        # 入力フォームエリア
        self.input_frame = ttk.LabelFrame(self.prediction_tab, text="レース情報入力")
        self.input_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 入力フォームの作成
        self.create_input_fields()
        
        # 予測ボタン
        self.predict_button = ttk.Button(
            self.prediction_tab, 
            text="AI予想を実行", 
            command=self.run_prediction
        )
        self.predict_button.pack(pady=10)
        
        # 結果表示エリア
        self.results_frame = ttk.LabelFrame(self.prediction_tab, text="予想結果")
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 結果テキストエリア
        self.results_text = tk.Text(self.results_frame, height=15)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)

    def create_input_fields(self):
        """入力フィールドを作成する"""

        # URL入力フィールド（レース情報をWebから取得するため）
        ttk.Label(self.input_frame, text="レースURL:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        self.url_entry = ttk.Entry(self.input_frame, width=30)
        self.url_entry.grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)

        # 情報取得ボタン
        self.fetch_button = ttk.Button(
            self.input_frame, 
            text="情報取得", 
            command=self.fetch_race_info
        )
        self.fetch_button.grid(row=0, column=6, sticky=tk.W, padx=5, pady=2)

        # レース基本情報
        ttk.Label(self.input_frame, text="レース名:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.race_name_entry = ttk.Entry(self.input_frame, width=30)
        self.race_name_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(self.input_frame, text="レース距離:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.race_distance_entry = ttk.Entry(self.input_frame, width=10)
        self.race_distance_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # 馬場状態と天候の入力フィールド
        ttk.Label(self.input_frame, text="馬場状態:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.track_condition = tk.StringVar()
        self.track_condition_combobox = ttk.Combobox(self.input_frame, width=10, textvariable=self.track_condition, values=["良", "稍重", "重", "不良"])
        self.track_condition_combobox.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)

        ttk.Label(self.input_frame, text="天候:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.weather = tk.StringVar()
        self.weather_combobox = ttk.Combobox(self.input_frame, width=10, textvariable=self.weather, values=["晴", "曇", "雨", "雪"])
        self.weather_combobox.grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)

        # 馬情報入力（複数行）
        self.horse_entries = []  # ここで明示的にリストを初期化

        headers = ["馬名", "騎手名", "枠番", "馬番", "人気", "馬体重", "体重増減"]
        for i, header in enumerate(headers):
            ttk.Label(self.input_frame, text=header).grid(row=2, column=i, padx=5, pady=2)

        # 最大10頭分の入力欄を作成
        for i in range(10):
            horse_row = []
            for j in range(len(headers)):
                entry = ttk.Entry(self.input_frame, width=10)
                # 枠番と馬番を自動入力
                if j == 2:  # 枠番
                    entry.insert(0, str(i+1))
                elif j == 3:  # 馬番
                    entry.insert(0, str(i+1))
                entry.grid(row=i+3, column=j, padx=5, pady=2)
                horse_row.append(entry)

            self.horse_entries.append(horse_row)  # ここでしっかりリストに追加

        print(f"✅ {len(self.horse_entries)} 頭分のエントリーを作成しました")

    
    def setup_history_tab(self):
        """履歴タブのコンポーネントをセットアップする"""
        # 履歴リストビュー
        self.history_frame = ttk.Frame(self.history_tab)
        self.history_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 列の定義
        columns = ("date", "race_name", "accuracy")
        self.history_tree = ttk.Treeview(self.history_frame, columns=columns, show="headings")
        
        # 列見出しの設定
        self.history_tree.heading("date", text="日付")
        self.history_tree.heading("race_name", text="レース名")
        self.history_tree.heading("accuracy", text="的中率")
        
        # 列幅の設定
        self.history_tree.column("date", width=100)
        self.history_tree.column("race_name", width=200)
        self.history_tree.column("accuracy", width=100)
        
        # スクロールバーの追加
        scrollbar = ttk.Scrollbar(self.history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscroll=scrollbar.set)
        
        # ウィジェットの配置
        self.history_tree.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_settings_tab(self):
        """設定タブのコンポーネントをセットアップする"""
        # 設定フレーム
        self.settings_frame = ttk.Frame(self.settings_tab)
        self.settings_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # API設定セクション
        api_frame = ttk.LabelFrame(self.settings_frame, text="Anthropic API設定")
        api_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(api_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.api_key_entry = ttk.Entry(api_frame, width=40, show="*")
        self.api_key_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # モデル設定セクション
        model_frame = ttk.LabelFrame(self.settings_frame, text="モデル設定")
        model_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(model_frame, text="使用モデル:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.model_path_entry = ttk.Entry(model_frame, width=40)
        self.model_path_entry.insert(0, self.model_path)
        self.model_path_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 保存ボタン
        save_button = ttk.Button(self.settings_frame, text="設定を保存", command=self.save_settings)
        save_button.pack(pady=10)
        
    def populate_form_with_race_info(self, race_info):
        print("フォームに情報を設定します...")

        # レース名と距離をセット
        self.race_name_entry.delete(0, tk.END)
        self.race_name_entry.insert(0, race_info.get('race_name', ''))

        self.race_distance_entry.delete(0, tk.END)
        self.race_distance_entry.insert(0, str(race_info.get('distance', '')))

        # 天候と馬場状態をセット
        print(f"天候: {race_info.get('weather', '不明')}, 馬場状態: {race_info.get('track_condition', '不明')}")
        self.weather.set(race_info.get('weather', '不明'))
        self.track_condition.set(race_info.get('track_condition', '不明'))

        # ここで self.horse_entries の内容を確認
        print(f"self.horse_entries の数: {len(self.horse_entries)}")

        if not self.horse_entries:
            print("❌ self.horse_entries が空です！フォームが正しく作成されていない可能性があります。")
            return

        # 馬データを入力
        for i, horse in enumerate(race_info['horses']):
            if i >= len(self.horse_entries):
                print(f"⚠️ 馬 {horse['horse_name']} のデータを入れる Entry が足りません。")
                continue

            entries = self.horse_entries[i]

            # 各エントリをクリア
            for entry in entries:
                entry.delete(0, tk.END)

            print(f"✅ {i+1}番目の馬 {horse['horse_name']} を入力中")

            # 各項目にデータをセット
            try:
                entries[0].insert(0, horse['horse_name'])
                entries[1].insert(0, horse['jockey'])
                entries[2].insert(0, str(horse['frame_number']))
                entries[3].insert(0, str(horse['horse_number']))
                entries[4].insert(0, str(horse['popularity']))
                entries[5].insert(0, str(horse['horse_weight']))
                weight_change = horse.get('weight_change', 0)
                weight_direction = horse.get('weight_direction', 1)
                weight_sign = "-" if weight_direction < 0 else "+"
                entries[6].insert(0, f"{weight_sign}{abs(weight_change)}")

            except Exception as e:
                print(f"❌ エラー: {str(e)}")


                # 体重増減の処理
                weight_change = horse.get('weight_change', 0)
                weight_direction = horse.get('weight_direction', 1)
                weight_sign = "-" if weight_direction < 0 else "+"
                entries[6].insert(0, f"{weight_sign}{abs(weight_change)}")

    def fetch_race_info(self):
        """URLからレース情報を取得してフォームに設定する"""
        url = self.url_entry.get()

        if not url:
            import tkinter.messagebox as messagebox
            messagebox.showwarning("URL入力", "レースページのURLを入力してください")
            return

        import tkinter.messagebox as messagebox
        messagebox.showinfo("情報取得", "レース情報を取得中です...")

        try:
            from src.utils.scraper import scrape_race_page
            race_info = scrape_race_page(url)

            if not race_info or not race_info.get('race_name'):
                messagebox.showerror("情報取得エラー", "レース情報を取得できませんでした")
                return

            # 取得データをデバッグ出力
            print(f"取得したレース情報: {race_info}")

            # 取得した情報をフォームに設定
            self.populate_form_with_race_info(race_info)
            messagebox.showinfo("情報取得", f"{len(race_info['horses'])}頭の情報を取得しました！")

        except Exception as e:
            import traceback
            traceback.print_exc()  # エラーの詳細をコンソールに表示
            messagebox.showerror("情報取得エラー", f"レース情報の取得中にエラーが発生しました: {str(e)}")

            
    def collect_horse_data(self):
        """入力フォームから馬データを収集する"""
        horse_data = []
        
        for entries in self.horse_entries:
            # 空の行はスキップ
            if not entries[0].get().strip():
                continue
                
            horse = {
                "horse_name": entries[0].get().strip(),
                "jockey": entries[1].get().strip(),
                "frame_number": entries[2].get().strip(),
                "horse_number": entries[3].get().strip(),
                "popularity": entries[4].get().strip(),
                "horse_weight": entries[5].get().strip(),
                "weight_change": entries[6].get().strip()
            }
            
            # 数値に変換できる項目は変換
            for key in ["frame_number", "horse_number", "popularity", "horse_weight", "weight_change"]:
                if horse[key] and horse[key].isdigit():
                    horse[key] = int(horse[key])
            
            horse_data.append(horse)
        
        return horse_data
    
    def run_prediction(self):
        """予測を実行する"""
        # 入力データの収集
        race_name = self.race_name_entry.get()
        race_distance = int(self.race_distance_entry.get()) if self.race_distance_entry.get() else None
        track_condition = self.track_condition.get() if hasattr(self, 'track_condition') else ""
        weather = self.weather.get() if hasattr(self, 'weather') else ""
        horse_data = self.collect_horse_data()
        
        if not race_name or not race_distance or not horse_data:
            import tkinter.messagebox as messagebox
            messagebox.showerror("入力エラー", "必要な情報を入力してください")
            return
        
        # 予測の実行（まだモデルがなければダミー予測を行う）
        try:
            race_info = {
                "race_name": race_name,
                "distance": race_distance,
                "track_condition": track_condition,
                "weather": weather,
                "horses": horse_data
            }
            
            # 予測処理
            predictions = self.predict_race(race_info)
            
            # Anthropic APIによる解説生成
            from src.anthropic.client import AnthropicClient
            client = AnthropicClient()
            explanation = client.generate_prediction_explanation(race_info, predictions)
            
            # 結果の表示
            self.display_results(predictions, explanation)
            
            # 履歴に追加
            self.add_to_history(race_name, predictions)
            
        except Exception as e:
            import tkinter.messagebox as messagebox
            messagebox.showerror("予測エラー", f"予測処理中にエラーが発生しました: {str(e)}")
    
    def predict_race(self, race_info):
        """レース予測を行う（モデルがなければダミー予測を返す）"""
        # 辞書から必要な情報を取り出す
        race_name = race_info["race_name"]
        race_distance = race_info["distance"]
        horse_data = race_info["horses"]
        
        # ダミー予測（テスト用）
        import random
        for horse in horse_data:
            # 人気順に近い予想を作成（実際のモデルの代わり）
            base_rank = horse.get("popularity", random.randint(1, len(horse_data)))
            if isinstance(base_rank, str):
                base_rank = random.randint(1, len(horse_data))
            # ちょっとランダム性を加える
            predicted_rank = max(1, min(len(horse_data), base_rank + random.randint(-2, 2)))
            horse["predicted_rank"] = predicted_rank
            horse["confidence"] = max(30, 100 - (predicted_rank * 10))
        
        # 予測順位でソート
        sorted_horses = sorted(horse_data, key=lambda x: x["predicted_rank"])
        
        # 整形した結果を返す
        return {
            "race_name": race_name,
            "distance": race_distance,
            "ranked_horses": sorted_horses
        }
    
    def display_results(self, predictions, explanation):
        """予測結果と解説を表示する"""
        # 結果テキストエリアをクリア
        self.results_text.delete(1.0, tk.END)
        
        # 予測結果を表示
        self.results_text.insert(tk.END, f"【{predictions['race_name']} ({predictions['distance']}m) AI予想結果】\n\n")
        
        # 予測順位順に馬を表示
        for i, horse in enumerate(predictions["ranked_horses"]):
            confidence = horse.get("confidence", 50)
            confidence_stars = "★" * int(confidence/20)  # 信頼度を★で表現
            self.results_text.insert(tk.END, f"{i+1}着予想: {horse['horse_name']} (騎手: {horse['jockey']})\n")
            self.results_text.insert(tk.END, f"    信頼度: {confidence}% {confidence_stars}\n")
        
        # Anthropicの解説を表示
        self.results_text.insert(tk.END, "\n【解説】\n\n")
        self.results_text.insert(tk.END, explanation)
    
    def add_to_history(self, race_name, predictions):
        """予測履歴に追加する"""
        date = datetime.now().strftime("%Y-%m-%d")
        accuracy = "予測済"  # 実際の的中率はレース後に更新
        
        # 履歴リストに追加
        self.history_tree.insert("", tk.END, values=(date, race_name, accuracy))
        
        # 履歴をファイルに保存
        self.save_prediction_history(race_name, predictions)
    
    def save_prediction_history(self, race_name, predictions):
        """予測履歴をファイルに保存する"""
        history_dir = "data/history"
        os.makedirs(history_dir, exist_ok=True)
        
        date = datetime.now().strftime("%Y%m%d")
        filename = f"{history_dir}/{date}_{race_name.replace(' ', '_')}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    def save_settings(self):
        """設定を保存する"""
        api_key = self.api_key_entry.get()
        model_path = self.model_path_entry.get()
        
        # 設定をファイルに保存
        config_dir = "config"
        os.makedirs(config_dir, exist_ok=True)
        
        settings = {
            "api_key": api_key,
            "model_path": model_path
        }
        
        with open(f"{config_dir}/settings.json", "w") as f:
            json.dump(settings, f)
        
        # 設定の更新
        if model_path != self.model_path:
            self.model_path = model_path
            self.load_model()
        
        import tkinter.messagebox as messagebox
        messagebox.showinfo("設定保存", "設定が保存されました")

def main():
    root = tk.Tk()
    app = BaneiAIApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()