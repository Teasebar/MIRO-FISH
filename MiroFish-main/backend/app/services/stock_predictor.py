"""
股票价格预测服务
使用技术分析指标和LLM分析预测股票走势，给出持有/卖出建议
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger('mirofish.services.stock_predictor')


def _try_import_yfinance():
    try:
        import yfinance as yf
        return yf
    except ImportError:
        raise ImportError(
            "yfinance is required for stock prediction. "
            "Install it with: pip install yfinance"
        )


class StockDataFetcher:
    """获取股票历史数据和基本面信息"""

    def fetch(self, ticker: str, period: str = "3mo") -> Dict[str, Any]:
        """
        获取股票数据

        Args:
            ticker: 股票代码（如 "SLIP"）
            period: 数据周期，如 "1mo", "3mo", "6mo", "1y"

        Returns:
            包含价格历史、技术指标和基本面的字典
        """
        yf = _try_import_yfinance()
        stock = yf.Ticker(ticker)

        hist = stock.history(period=period)
        if hist.empty:
            raise ValueError(f"无法获取股票 {ticker} 的历史数据，请确认代码正确")

        prices = hist["Close"].tolist()
        volumes = hist["Volume"].tolist()
        dates = [str(d.date()) for d in hist.index]

        indicators = self._compute_indicators(prices, volumes)

        info = {}
        try:
            raw_info = stock.info or {}
            for key in [
                "shortName", "sector", "industry",
                "marketCap", "trailingPE", "forwardPE",
                "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
                "fiftyDayAverage", "twoHundredDayAverage",
                "dividendYield", "beta", "shortRatio",
                "recommendationKey", "targetMeanPrice",
            ]:
                val = raw_info.get(key)
                if val is not None:
                    info[key] = val
        except Exception as e:
            logger.warning(f"获取 {ticker} 基本面信息失败: {e}")

        return {
            "ticker": ticker.upper(),
            "period": period,
            "dates": dates,
            "prices": [round(p, 4) for p in prices],
            "volumes": volumes,
            "current_price": round(prices[-1], 4),
            "price_change_pct": round(
                (prices[-1] - prices[0]) / prices[0] * 100, 2
            ) if prices[0] else 0,
            "indicators": indicators,
            "info": info,
        }

    def _compute_indicators(
        self, prices: List[float], volumes: List[int]
    ) -> Dict[str, Any]:
        """计算常用技术指标"""
        indicators: Dict[str, Any] = {}

        def sma(data, n):
            if len(data) < n:
                return None
            return round(sum(data[-n:]) / n, 4)

        indicators["sma_5"] = sma(prices, 5)
        indicators["sma_20"] = sma(prices, 20)
        indicators["sma_50"] = sma(prices, 50)

        # RSI (14)
        indicators["rsi_14"] = self._rsi(prices, 14)

        # MACD (12, 26, 9)
        macd_line, signal_line, histogram = self._macd(prices)
        indicators["macd"] = {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
        }

        # Bollinger Bands (20, 2)
        bb = self._bollinger_bands(prices, 20, 2)
        indicators["bollinger_bands"] = bb

        # Volume trend (avg last 5 vs avg last 20)
        if len(volumes) >= 20:
            avg_vol_5 = sum(volumes[-5:]) / 5
            avg_vol_20 = sum(volumes[-20:]) / 20
            indicators["volume_trend"] = round(
                (avg_vol_5 - avg_vol_20) / avg_vol_20 * 100, 2
            )
        else:
            indicators["volume_trend"] = None

        return indicators

    def _ema(self, prices: List[float], n: int) -> Optional[float]:
        if len(prices) < n:
            return None
        k = 2 / (n + 1)
        ema = sum(prices[:n]) / n
        for p in prices[n:]:
            ema = p * k + ema * (1 - k)
        return round(ema, 4)

    def _rsi(self, prices: List[float], n: int = 14) -> Optional[float]:
        if len(prices) < n + 1:
            return None
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains[:n]) / n
        avg_loss = sum(losses[:n]) / n
        for i in range(n, len(deltas)):
            avg_gain = (avg_gain * (n - 1) + gains[i]) / n
            avg_loss = (avg_loss * (n - 1) + losses[i]) / n
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - 100 / (1 + rs), 2)

    def _macd(
        self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if len(prices) < slow + signal:
            return None, None, None
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        if ema_fast is None or ema_slow is None:
            return None, None, None
        macd_line = round(ema_fast - ema_slow, 4)

        # Build a macd series for the signal EMA
        macd_series = []
        k_fast = 2 / (fast + 1)
        k_slow = 2 / (slow + 1)
        ema_f = sum(prices[:fast]) / fast
        ema_s = sum(prices[:slow]) / slow
        for p in prices[slow:]:
            ema_f = p * k_fast + ema_f * (1 - k_fast)
            ema_s = p * k_slow + ema_s * (1 - k_slow)
            macd_series.append(ema_f - ema_s)

        if len(macd_series) < signal:
            return macd_line, None, None

        k_sig = 2 / (signal + 1)
        sig = sum(macd_series[:signal]) / signal
        for v in macd_series[signal:]:
            sig = v * k_sig + sig * (1 - k_sig)
        signal_line = round(sig, 4)
        histogram = round(macd_line - signal_line, 4)
        return macd_line, signal_line, histogram

    def _bollinger_bands(
        self, prices: List[float], n: int = 20, k: float = 2
    ) -> Optional[Dict[str, float]]:
        if len(prices) < n:
            return None
        window = prices[-n:]
        mid = sum(window) / n
        std = (sum((p - mid) ** 2 for p in window) / n) ** 0.5
        return {
            "upper": round(mid + k * std, 4),
            "middle": round(mid, 4),
            "lower": round(mid - k * std, 4),
        }


class StockPredictor:
    """使用LLM对股票数据进行综合分析，给出预测和建议"""

    def __init__(self):
        self.fetcher = StockDataFetcher()
        self.llm = LLMClient()

    def predict(self, ticker: str, period: str = "3mo") -> Dict[str, Any]:
        """
        预测股票走势并给出持有/卖出建议

        Args:
            ticker: 股票代码
            period: 分析历史数据周期

        Returns:
            包含预测结果和建议的字典
        """
        logger.info(f"开始预测股票 {ticker} ({period})")
        data = self.fetcher.fetch(ticker, period)
        analysis = self._analyze(data)
        logger.info(f"股票 {ticker} 分析完成: {analysis.get('recommendation')}")
        return {
            "ticker": data["ticker"],
            "current_price": data["current_price"],
            "price_change_pct": data["price_change_pct"],
            "period_analyzed": period,
            "indicators": data["indicators"],
            "info": data["info"],
            "analysis": analysis,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    def _analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """调用LLM进行综合分析"""
        prompt = self._build_prompt(data)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional quantitative analyst and financial advisor. "
                    "Analyze the provided stock data and technical indicators, then give "
                    "a clear, evidence-based recommendation. "
                    "Always respond in valid JSON format."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            result = self.llm.chat_json(messages, temperature=0.3, max_tokens=1024)
        except Exception as e:
            logger.error(f"LLM分析失败: {e}")
            result = self._rule_based_fallback(data)

        # Normalise recommendation to uppercase
        rec = result.get("recommendation", "HOLD").upper()
        if rec not in ("HOLD", "SELL", "BUY"):
            rec = "HOLD"
        result["recommendation"] = rec
        return result

    def _build_prompt(self, data: Dict[str, Any]) -> str:
        ind = data["indicators"]
        info = data["info"]
        bb = ind.get("bollinger_bands") or {}

        lines = [
            f"Stock: {data['ticker']}",
            f"Current price: ${data['current_price']}",
            f"Price change over period ({data['period']}): {data['price_change_pct']}%",
        ]

        if info.get("shortName"):
            lines.append(f"Company: {info['shortName']}")
        if info.get("sector"):
            lines.append(f"Sector: {info['sector']}")
        if info.get("fiftyTwoWeekHigh") and info.get("fiftyTwoWeekLow"):
            lines.append(
                f"52-week range: ${info['fiftyTwoWeekLow']} – ${info['fiftyTwoWeekHigh']}"
            )
        if info.get("targetMeanPrice"):
            lines.append(f"Analyst target price: ${info['targetMeanPrice']}")
        if info.get("recommendationKey"):
            lines.append(f"Analyst consensus: {info['recommendationKey']}")
        if info.get("trailingPE"):
            lines.append(f"Trailing P/E: {info['trailingPE']}")
        if info.get("beta"):
            lines.append(f"Beta: {info['beta']}")

        lines += [
            "",
            "Technical indicators:",
            f"  SMA-5:  {ind.get('sma_5')}",
            f"  SMA-20: {ind.get('sma_20')}",
            f"  SMA-50: {ind.get('sma_50')}",
            f"  RSI-14: {ind.get('rsi_14')} (overbought >70, oversold <30)",
        ]

        macd = ind.get("macd") or {}
        lines += [
            f"  MACD line:   {macd.get('macd_line')}",
            f"  Signal line: {macd.get('signal_line')}",
            f"  Histogram:   {macd.get('histogram')} (positive = bullish momentum)",
        ]

        if bb:
            lines += [
                f"  Bollinger Upper:  {bb.get('upper')}",
                f"  Bollinger Middle: {bb.get('middle')}",
                f"  Bollinger Lower:  {bb.get('lower')}",
            ]

        vol_trend = ind.get("volume_trend")
        if vol_trend is not None:
            lines.append(
                f"  Volume trend (5d vs 20d avg): {vol_trend:+.2f}%"
                + (" (above average)" if vol_trend > 0 else " (below average)")
            )

        lines += [
            "",
            "Based on the above data, provide a JSON response with the following fields:",
            '  "recommendation": "HOLD" | "SELL" | "BUY"',
            '  "confidence": a number between 0 and 1',
            '  "short_term_outlook": one sentence on price direction in the next 1-2 weeks',
            '  "key_signals": list of 3-5 most important technical/fundamental signals observed',
            '  "risks": list of 2-3 main risks to your recommendation',
            '  "reasoning": 2-3 sentences explaining the recommendation',
        ]

        return "\n".join(lines)

    def _rule_based_fallback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """当LLM不可用时的规则回退"""
        ind = data["indicators"]
        rsi = ind.get("rsi_14")
        macd = (ind.get("macd") or {}).get("histogram")
        price = data["current_price"]
        sma20 = ind.get("sma_20")

        bullish = 0
        bearish = 0

        if rsi is not None:
            if rsi < 30:
                bullish += 2
            elif rsi > 70:
                bearish += 2

        if macd is not None:
            if macd > 0:
                bullish += 1
            else:
                bearish += 1

        if sma20 is not None:
            if price > sma20:
                bullish += 1
            else:
                bearish += 1

        if bullish > bearish:
            rec = "BUY"
        elif bearish > bullish:
            rec = "SELL"
        else:
            rec = "HOLD"

        return {
            "recommendation": rec,
            "confidence": 0.5,
            "short_term_outlook": "Rule-based analysis only; LLM unavailable.",
            "key_signals": [
                f"RSI-14: {rsi}",
                f"MACD histogram: {macd}",
                f"Price vs SMA-20: {'above' if sma20 and price > sma20 else 'below'}",
            ],
            "risks": ["LLM analysis unavailable", "Rule-based signals only"],
            "reasoning": (
                f"Rule-based scoring: bullish={bullish}, bearish={bearish}. "
                f"Recommendation based on RSI, MACD, and price vs SMA-20."
            ),
        }
