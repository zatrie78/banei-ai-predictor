import requests
from bs4 import BeautifulSoup
import re
import time
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def scrape_race_page(url):
    logger = setup_logging()
    logger.info(f"URLからレース情報を取得中: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    time.sleep(1)  # サーバー負荷を考慮
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # レース名の取得
        race_header = soup.select_one('.race_header h1, .RaceName')
        race_name = race_header.text.strip() if race_header else "不明なレース"
        logger.info(f"レース名: {race_name}")
        
        # レース距離の取得
        course_text = ""
        course_element = soup.select_one('div.Courseコース詳細, div.RaceData01, .RaceData')
        if course_element:
            course_text = course_element.text.strip()
        
        distance_match = re.search(r'(\d+)m', course_text)
        distance = int(distance_match.group(1)) if distance_match else 0
        logger.info(f"レース距離: {distance}m")
        
        # 天候と馬場状態
        track_condition = ""
        weather = ""
        
        track_elem = soup.select_one('.RaceData01 .Item03, .RaceData .Item03')
        if track_elem:
            track_text = track_elem.text.strip()
            track_parts = track_text.split('/')
            if len(track_parts) >= 2:
                weather = track_parts[0].strip()
                track_condition = track_parts[1].strip()
        logger.info(f"天候: {weather}, 馬場状態: {track_condition}")
        
        horses = []
        horse_rows = soup.select('table.RaceTable01 tr:not(.Header), table.Shutuba_Table tr.HorseList')
        logger.info(f"出走馬数: {len(horse_rows)}頭")
        
        for i, row in enumerate(horse_rows):
            tds = row.select('td')  # すべての <td> を取得
            if len(tds) < 11:  # 必要なデータがそろっていない場合はスキップ
                continue

            try:
                frame_number = tds[0].text.strip() if len(tds) > 0 else "0"
                horse_number = tds[1].text.strip() if len(tds) > 1 else "0"

                # --- **馬名の取得 (最初のコードのロジックに戻す！)** ---
                horse_name_td = row.select_one('td.Horse_Name, .HorseInfo a')  # 最初の正しいセレクタを使用
                horse_name = horse_name_td.text.strip() if horse_name_td else "不明"

                # --- **騎手名の取得 (修正: 正しいカラム)** ---
                jockey = "不明"
                if len(tds) > 6:  # 騎手のデータがあるか確認
                    jockey = tds[6].text.strip()  # 7番目の<td>が騎手名
                
                # --- **馬体重の取得 (修正)** ---
                horse_weight = "0"
                weight_match = re.search(r'(\d+)', tds[8].text.strip())  # 9番目の<td>
                if weight_match:
                    horse_weight = weight_match.group(1)

                # --- **オッズの取得 (修正)** ---
                odds = "0.0"
                odds_text = tds[9].text.strip().replace(',', '.')  # 10番目の<td>
                odds_match = re.search(r'\d+\.\d+', odds_text)
                if odds_match:
                    odds = odds_match.group(0)

                # --- **人気の取得 (修正)** ---
                popularity = "0"
                popularity_match = re.search(r'\d+', tds[10].text.strip())  # 11番目の<td>
                if popularity_match:
                    popularity = popularity_match.group(0)

                # --- **馬体重の増減の取得 (修正: 元のコードに戻す)** ---
                weight_change = "0"
                weight_change_td = tds[8].select_one('small')  # <small> の中に増減がある
                if weight_change_td:
                    weight_change_match = re.search(r'[-+]?\d+', weight_change_td.text.strip())
                    if weight_change_match:
                        weight_change = weight_change_match.group(0)

                # デバッグ用ログ出力
                logger.info(f"{horse_name} (騎手: {jockey}, 枠番: {frame_number}, 馬番: {horse_number}) - オッズ: {odds}, 人気: {popularity}, 馬体重: {horse_weight}kg, 増減: {weight_change}")
                
                horses.append({
                    'frame_number': int(frame_number) if frame_number.isdigit() else 0,
                    'horse_number': int(horse_number) if horse_number.isdigit() else 0,
                    'horse_name': horse_name,
                    'jockey': jockey,
                    'odds': float(odds),
                    'popularity': int(popularity),
                    'horse_weight': int(horse_weight),
                    'weight_change': int(weight_change),
                    'weight_direction': 1 if weight_change.startswith('+') else (-1 if weight_change.startswith('-') else 0)
                })
            
            except Exception as e:
                logger.error(f"馬{i+1}の情報取得エラー: {str(e)}")
        
        result = {
            'race_name': race_name,
            'distance': distance,
            'track_condition': track_condition,
            'weather': weather,
            'horses': horses
        }
        
        logger.info(f"スクレイピング完了: {len(horses)}頭の情報を取得")
        return result

    except requests.RequestException as e:
        logger.error(f"リクエストエラー: {str(e)}")
        return None


# 単体テスト用
if __name__ == "__main__":
    # テスト用URL
    test_url = "https://nar.netkeiba.com/race/shutuba.html?race_id=202065031608"  # 実際のURLに変更してください
    
    try:
        # ユーザー入力でURLを取得（テスト用）
        input_url = input("スクレイピングするレースページのURLを入力してください: ")
        if input_url:
            test_url = input_url
        
        # スクレイピング実行
        result = scrape_race_page(test_url)
        
        # 結果の出力
        print("\n--- スクレイピング結果 ---")
        print(f"レース名: {result['race_name']}")
        print(f"距離: {result['distance']}m")
        print(f"馬場状態: {result['track_condition']}")
        print(f"天候: {result['weather']}")
        print(f"出走頭数: {len(result['horses'])}頭")
        
        print("\n出走馬情報:")
        for i, horse in enumerate(result['horses']):
            print(f"{i+1}. {horse['horse_name']} (騎手: {horse['jockey']}, 馬番: {horse['horse_number']}, 枠番: {horse['frame_number']})")
            print(f"   オッズ: {horse['odds']}, 人気: {horse['popularity']}, 馬体重: {horse['horse_weight']}kg({'+' if horse['weight_direction'] > 0 else ''}{-horse['weight_change'] if horse['weight_direction'] < 0 else horse['weight_change']})")
    
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")