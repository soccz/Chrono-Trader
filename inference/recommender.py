import pandas as pd
import numpy as np
import os
from datetime import datetime

from utils.config import config
from utils.logger import logger

def run(predictions: list):
    """Analyzes predictions and uncertainty to generate user-friendly recommendations."""
    logger.info("=== Starting Recommendation Generation (Ensembled + User-Friendly) ===")
    
    if not predictions:
        logger.warning("Recommender received no predictions to analyze.")
        return [] # Ensure it returns an empty list if no predictions

    potential_trades = []
    for pred in predictions:
        total_change = np.sum(pred['predicted_pattern'])
        confidence = 1 / (1 + pred['uncertainty'])

        potential_trades.append({
            'market': pred['market'],
            'potential': total_change,
            'pattern': pred['predicted_pattern'],
            'confidence': confidence,
            'current_price': pred['current_price'],
            'strategy': pred.get('strategy', 'trending') # Default to 'trending' for safety
        })

    # Separate by strategy
    trending_trades = [t for t in potential_trades if t['strategy'] == 'trending']
    pattern_trades = [t for t in potential_trades if t['strategy'] == 'pattern']

    # Sort trending trades by confidence and take top 5
    trending_trades.sort(key=lambda x: x['confidence'], reverse=True)
    top_trending = trending_trades[:5]

    # Pattern trades are already the top 3, so just sort them by confidence
    pattern_trades.sort(key=lambda x: x['confidence'], reverse=True)

    # Combine and create the final list for display
    top_trades = top_trending + pattern_trades
    top_trades.sort(key=lambda x: (x['confidence'], abs(x['potential'])), reverse=True)


    df_to_save = []
    logger.info(f"\n=== 상세 암호화폐 거래 추천 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===")
    logger.info(f"분석된 유망 코인 수: {len(predictions)}개")
    logger.info(f"추천 개수: {len(top_trades)}개") # This will now reflect the actual number

    for i, trade in enumerate(top_trades):
        signal = "Long (매수)" if trade['potential'] > 0 else "Short (매도)"
        entry_price = trade['current_price']

        # --- 데이터 프레임 저장용 데이터 준비 ---
        trade_data = trade.copy()
        trade_data['signal'] = "Long" if trade['potential'] > 0 else "Short"
        df_to_save.append(trade_data)

        # --- 사용자 친화적 로그 출력 ---
        logger.info(f"\n--- {i+1}. {trade['market']} ---")
        logger.info(f"*   추천 신호: {signal}")
        logger.info(f"*   현재 가격: {entry_price:,.0f}원")
        logger.info(f"*   신뢰도: {trade['confidence']:.2%}")
        logger.info(f"*   예상 수익률 (6시간 합산): {trade['potential']:.2%}")
        logger.info("*   시간별 예상 등락률:")
        for hour, p_val in enumerate(trade['pattern']):
            logger.info(f"    *   {hour+1}시간 후: {p_val:+.2%}")

    # Save the recommendations to a CSV file
    if df_to_save:
        df = pd.DataFrame(df_to_save)
        
        # Format the pattern array for better CSV readability
        df['pattern'] = df['pattern'].apply(lambda p: ','.join([f'{x:.4f}' for x in p]))

        # Reorder columns for clarity
        df = df[['market', 'signal', 'potential', 'confidence', 'current_price', 'pattern']]

        # Create directory if it doesn't exist
        output_dir = 'recommendations'
        os.makedirs(output_dir, exist_ok=True)

        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"recs_{timestamp}.csv")
        
        try:
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"Recommendations successfully saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save recommendations to CSV: {e}")

    logger.info("======================================================")
    return df_to_save