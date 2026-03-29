"""
股票预测 API 路由
提供基于技术分析 + LLM 的股票持有/卖出建议

Usage:
    GET /api/stock/predict/SLIP
    GET /api/stock/predict/<ticker>?period=3mo
"""

import traceback
from flask import request, jsonify

from . import stock_bp
from ..services.stock_predictor import StockPredictor
from ..utils.logger import get_logger

logger = get_logger('mirofish.api.stock')

_predictor = None


def _get_predictor() -> StockPredictor:
    global _predictor
    if _predictor is None:
        _predictor = StockPredictor()
    return _predictor


@stock_bp.route('/predict/<ticker>', methods=['GET'])
def predict_stock(ticker: str):
    """
    预测指定股票的价格走势，返回持有/卖出/买入建议

    路径参数:
        ticker  股票代码，例如 SLIP

    查询参数:
        period  历史数据周期（默认 3mo）。可选值: 1mo, 3mo, 6mo, 1y

    返回:
        {
            "success": true,
            "data": {
                "ticker": "SLIP",
                "current_price": 4.32,
                "price_change_pct": -12.5,
                "period_analyzed": "3mo",
                "indicators": { ... },
                "info": { ... },
                "analysis": {
                    "recommendation": "HOLD",   // HOLD | SELL | BUY
                    "confidence": 0.72,
                    "short_term_outlook": "...",
                    "key_signals": [...],
                    "risks": [...],
                    "reasoning": "..."
                },
                "generated_at": "2026-03-29T00:00:00Z"
            }
        }
    """
    ticker = ticker.strip().upper()
    if not ticker or not ticker.replace('.', '').replace('-', '').isalnum():
        return jsonify({
            "success": False,
            "error": f"无效的股票代码: {ticker}"
        }), 400

    period = request.args.get('period', '3mo')
    valid_periods = {'1mo', '3mo', '6mo', '1y'}
    if period not in valid_periods:
        return jsonify({
            "success": False,
            "error": f"period 参数无效，可选值: {', '.join(sorted(valid_periods))}"
        }), 400

    try:
        predictor = _get_predictor()
        result = predictor.predict(ticker, period)
        return jsonify({
            "success": True,
            "data": result
        })
    except ImportError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404
    except Exception as e:
        logger.error(f"股票预测失败 [{ticker}]: {e}\n{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@stock_bp.route('/predict', methods=['GET'])
def predict_slip_default():
    """
    默认预测 SLIP 股票（GET /api/stock/predict 的快捷方式）

    查询参数:
        period  历史数据周期（默认 3mo）
    """
    return predict_stock('SLIP')
