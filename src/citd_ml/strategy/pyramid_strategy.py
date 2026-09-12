import numpy as np
import pandas as pd


# ------------------------------------------------------------- chi bao
def momentum(close, period):
    """Momentum = close / close[period] * 100"""
    out = np.full(len(close), np.nan)
    out[period:] = close[period:] / close[:-period] * 100.0
    return out


def vwap(high, low, close, volume, period):
    """VWAP truot: tong(gia_dien_hinh * volume) / tong(volume) tren `period` bar."""
    typical = (high + low + close) / 3.0
    num = pd.Series(typical * volume).rolling(period).sum().values
    den = pd.Series(volume).rolling(period).sum().values
    return num / den


def atr(high, low, close, period):
    """ATR = trung binh truot cua True Range."""
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.Series(tr).rolling(period).mean().values


# ------------------------------------------------------------- vi the
class Position:
    """
    Mot vi the Long doc lap.

    Lenh goc va lenh nhoi dung chung lop nay - khong co khac biet nao ve cach
    quan ly. Chi khac gia vao, ATR dong bang, va do la he qua cua viec chung
    vao o cac bar khac nhau.
    """

    def __init__(self, bar, price, atr_frozen, sl_pct, pt_pct,
                 origin_bar, leg):
        self.entry_bar = bar
        self.entry_price = price
        self.atr_frozen = atr_frozen
        self.stop = price * (1 - sl_pct)
        self.take_profit = price * (1 + pt_pct)
        self.peak = -np.inf
        self.origin_bar = origin_bar   # bar co tin hieu sinh ra gia dinh nay
        self.leg = leg                 # 0 = lenh goc, 1..k = lenh nhoi thu may

    def trail(self, atr_mult):
        """Dời trailing stop. Goi mot lan khi bar M15 moi mo cua."""
        if self.peak == -np.inf:
            return
        candidate = self.peak - atr_mult * self.atr_frozen
        if candidate > self.stop:
            self.stop = candidate


# ------------------------------------------------------------ chien luoc
class PyramidStrategy:
    """
    Vao lenh
        Tin hieu tai bar i:
            Momentum(14)[i-1] < Momentum(14)[i-2]
            VA  VWAP(142)[i-1] > VWAP(142)[i-2]
            VA  khong con lenh goc nao dang mo
        Khi tin hieu no: mo lenh goc tai bar i, va hen mo lenh nhoi tai
        bar i+1, i+2, ..., i+k. Cac lenh nhoi vao vo dieu kien.

    Stop (ap dung y het cho ca lenh goc lan lenh nhoi)
        Khoi tao : gia_vao_cua_chinh_no * (1 - 0.006)
        Moi bar mo cua: stop = max(stop, dinh_gia - 3.8 * ATR_dong_bang)
        ATR khoa tai bar truoc bar vao lenh cua chinh no.
        Stop chi di len, va giu nguyen trong suot bar.

    Thoat lenh
        Gia cham stop, hoac cham profit target 33.7%, hoac thu Sau 20:40.
        Moi vi the thoat doc lap.

    Chi Long.
    """

    def __init__(self, k=3, mom_period=14, vwap_period=142, atr_period=90,
                 atr_mult=3.8, sl_pct=0.006, pt_pct=0.337,
                 friday_exit_sec=74400, warmup=200):
        self.k = k
        self.mom_period = mom_period
        self.vwap_period = vwap_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.sl_pct = sl_pct
        self.pt_pct = pt_pct
        self.friday_exit_sec = friday_exit_sec
        self.warmup = warmup
        self.reset()

    def reset(self):
        self.positions = []      # cac vi the dang mo
        self.pending = []        # (bar_se_vao, origin_bar, leg) da hen truoc
        self.base_open = False   # co lenh goc nao dang mo khong

    # -------------------------------------------------------- chi bao
    def prepare(self, open_, high, low, close, volume):
        """Tinh truoc cac chi bao tren toan chuoi M15."""
        return {
            "open": open_,
            "close": close,
            "mom": momentum(close, self.mom_period),
            "vwap": vwap(high, low, close, volume, self.vwap_period),
            "atr": atr(high, low, close, self.atr_period),
        }

    # -------------------------------------------------------- vao lenh
    def entry_signal(self, i, ind):
        """Dieu kien tin hieu, danh gia tai thoi diem bar i mo cua."""
        if i < self.warmup or self.base_open:
            return False
        mom, vw, at = ind["mom"], ind["vwap"], ind["atr"]
        if np.isnan(mom[i - 2]) or np.isnan(vw[i - 2]) or np.isnan(at[i - 1]):
            return False
        return mom[i - 1] < mom[i - 2] and vw[i - 1] > vw[i - 2]

    def _open(self, i, ind, origin_bar, leg):
        """Mo mot vi the tai gia mo cua bar i."""
        if np.isnan(ind["atr"][i - 1]):
            return
        p = Position(i, ind["open"][i], ind["atr"][i - 1],
                     self.sl_pct, self.pt_pct, origin_bar, leg)
        self.positions.append(p)
        if leg == 0:
            self.base_open = True

    def open_bar(self, i, ind):
        """
        Xu ly moi viec can lam khi bar i mo cua:
        vao cac lenh da hen, kiem tra tin hieu moi, roi dời trailing.
        """
        # 1. cac lenh nhoi da hen tu truoc
        due = [x for x in self.pending if x[0] == i]
        self.pending = [x for x in self.pending if x[0] != i]
        for _, origin_bar, leg in due:
            self._open(i, ind, origin_bar, leg)

        # 2. tin hieu moi -> lenh goc + hen k lenh nhoi
        if self.entry_signal(i, ind):
            self._open(i, ind, origin_bar=i, leg=0)
            for leg in range(1, self.k + 1):
                self.pending.append((i + leg, i, leg))

        # 3. dời trailing cho cac vi the da mo tu truoc
        for p in self.positions:
            if p.entry_bar < i:
                p.trail(self.atr_mult)

    # -------------------------------------------------------- quan ly
    def scan_bar(self, m1_high, m1_low):
        """
        Duyet cac nen M1 ben trong bar M15 hien tai theo thu tu thoi gian.
        Tra ve danh sach (vi_the, ly_do, gia_thoat) cua cac lenh dong trong bar.
        """
        closed = []
        alive = list(self.positions)
        for h, l in zip(m1_high, m1_low):
            if not alive:
                break
            still = []
            for p in alive:
                if l <= p.stop:
                    closed.append((p, "Stop", p.stop))
                    continue
                if h > p.peak:
                    p.peak = h
                if h >= p.take_profit:
                    closed.append((p, "Target", p.take_profit))
                    continue
                still.append(p)
            alive = still
        self.positions = alive
        self._after_close(closed)
        return closed

    def friday_close(self, weekday, seconds, price):
        """Cat toan bo vi the con lai vao chieu thu Sau."""
        if not (weekday == 4 and seconds >= self.friday_exit_sec):
            return []
        closed = [(p, "Friday", price) for p in self.positions]
        self.positions = []
        self._after_close(closed)
        return closed

    def _after_close(self, closed):
        """Neu lenh goc vua dong, mo lai cong cho tin hieu tiep theo."""
        if any(p.leg == 0 for p, _, _ in closed):
            self.base_open = False
