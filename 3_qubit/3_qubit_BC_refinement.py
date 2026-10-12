
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence
import math
import numpy as np


@dataclass(frozen=True)
class Term:
    label: str
    B: float
    C: float
    omega: float


TERMS = (
    Term('XII', .72, .08, 1.0), Term('ZZI', .59, .06, 1.1),
    Term('IXX', .36, .04, 1.2), Term('YIZ', .29, .03, 1.3),
    Term('XYZ', .18, .02, 1.4), Term('ZXY', .145, .015, 1.5),
)
TERM_BY_LABEL = {x.label:x for x in TERMS}
_EB = .003752998658062161
_EC = .0037764361580621612


_RAW_INPUTS = {
    'XII': (.720098704530998, .079892487472177, 1.00390625, .023311286931797057),
    'ZZI': (.589904490064346, .060087171461676, 1.09765625, .022788675497802422),
    'IXX': (.359820506579943, .040101014968662, 1.20703125, .021853634290456444),
    'YIZ': (.290003866335360, .029990397519258, 1.31640625, .02103650175108073),
    'XYZ': (.179983683216123, .020012203930205, 1.41015625, .019714137485719745),
    'ZXY': (.145541715865592, .014635124010779, 1.48828125, .018581209659652584),
}
DEFAULT_INPUTS = {
    lab:dict(B_hat=b,C_hat=c,epsilon_B=_EB,epsilon_C=_EC,
             omega_hat=w,frequency_error_bound=ew,omega_min=1.,omega_max=1.5)
    for lab,(b,c,w,ew) in _RAW_INPUTS.items()
}


@dataclass(frozen=True)
class Config:
    target_B_error: float = 1e-4       # 要求返回区间相对于估计中心的最大半径
    target_C_error: float = 1e-4
    max_B_time: float = 1024.0        # 最长单次询问时间，不是总实验时间
    max_C_time: float = 64.0
    max_C_model_phase_error: float = .20  # theta 的误差预算（不是 2*theta）
    shots: Optional[int] = None       # 例如 200000；每个时间、每个正交测量的次数
    measurement_failure_probability: float = .01  # 全六项、两种系数联合预算
    extra_phase_error: float = 0.0    # 已论证的额外 theta 误差上界（每次测量）
    max_alias_candidates: int = 10000
    seed: int = 20261012


def normalize_input(record: Mapping) -> dict:
    """既接收固定输入，也接收上一版 omega 程序的原始结果记录。"""
    if 'design' in record:
        d = record['design']
        get = (lambda k:d[k]) if isinstance(d,Mapping) else (lambda k:getattr(d,k))
        values = {k:float(get(k)) for k in
                  ('B_hat','C_hat','epsilon_B','epsilon_C','omega_min','omega_max')}
        values.update(omega_hat=float(record['omega_hat']),
                      frequency_error_bound=float(record['frequency_error_bound']))
    else:
        values = {k:float(record[k]) for k in
                  ('B_hat','C_hat','epsilon_B','epsilon_C','omega_hat',
                   'frequency_error_bound','omega_min','omega_max')}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError('所有输入必须为有限数值。')
    if min(values['epsilon_B'],values['epsilon_C'],values['frequency_error_bound'])<0:
        raise ValueError('误差界必须非负。')
    wh, ew = values['omega_hat'], values['frequency_error_bound']
    wlo = max(values['omega_min'],wh-ew)
    whi = min(values['omega_max'],wh+ew)
    if wh<=0 or wlo<=0 or whi<wlo:
        raise ValueError('频率先验与估计区间不相容，或含非正频率。')
    values.update(omega_lower=wlo,omega_upper=whi,
                  omega_uncertainty=max(abs(wlo-wh),abs(whi-wh)),
                  C_abs_upper=abs(values['C_hat'])+values['epsilon_C'])
    return values


def systematic_phase_bound(kind: str, t: float, p: Mapping) -> float:
    """返回 theta（半 Ramsey 相位）的模型误差上界，非真误差。"""
    c, ew = p['C_abs_upper'],p['omega_uncertainty']
    if kind=='B':
        return c/p['omega_lower']*min(ew*t,1.)
    if kind=='C':
        coefficient_bias = ((np.pi/2-1)*c*ew/p['omega_hat']
                            + np.pi*c/12*(ew*t)**2)
        return 2*t/np.pi*coefficient_bias
    raise ValueError('kind 必须为 B 或 C。')


def synchronized_schedule(kind: str, p: Mapping, cfg: Config) -> list[int]:
    period = 2*np.pi/p['omega_hat']
    cap = cfg.max_B_time if kind=='B' else cfg.max_C_time
    max_m = int(np.floor(cap/period+1e-12))
    if kind=='C':
        # beta_C(t) 单调增加；整数周期二分选择，不使用真实频率。
        left,right=0,max_m
        while left<right:
            middle=(left+right+1)//2
            if systematic_phase_bound('C',middle*period,p)<=cfg.max_C_model_phase_error:
                left=middle
            else:
                right=middle-1
        max_m=left
    if max_m<1:
        return []
    periods=[]
    m=1
    while m<=max_m:
        periods.append(m)
        m*=2
    if periods[-1]!=max_m:
        periods.append(max_m)
    return periods


def square_wave_integrals(omega_hat: float, omega: float, m: int) -> tuple[float,float]:
    """真实模拟信号积分；脉冲时间只由已学习 omega_hat 决定。"""
    t=2*np.pi*m/omega_hat
    switches=(2*np.arange(2*m)+1)*np.pi/(2*omega_hat)
    edges=np.r_[0.,switches,t]
    signs=(-1.)**np.arange(len(edges)-1)
    g=float(np.dot(signs,np.diff(edges)))
    f=float(np.dot(signs,np.diff(np.sin(omega*edges)))/omega)
    return g,f


def simulate_ramsey_experiment(label: str, kind: str, m: int,
                               omega_hat: float, shots: Optional[int],
                               rng: np.random.Generator) -> dict:
    """可替换为实验接口。只返回测量期望/计数，不向估计器提供真实相位。"""
    term=TERM_BY_LABEL[label]
    t=2*np.pi*m/omega_hat
    if kind=='B':
        theta=term.B*t+term.C/term.omega*np.sin(term.omega*t)
    else:
        g,f=square_wave_integrals(omega_hat,term.omega,m)
        theta=term.B*g+term.C*f
    exact=np.array([np.cos(2*theta),np.sin(2*theta)])
    if shots is None:
        measured=exact
        counts=None
    else:
        counts=rng.binomial(shots,np.clip((1+exact)/2,0,1))
        measured=2*counts/shots-1
    return dict(R=float(measured[0]),S=float(measured[1]),counts_plus=counts)


def phase_from_measurement(measurement: Mapping, shots: Optional[int],
                           per_pair_failure: float) -> tuple[float,float]:
    phase=.5*math.atan2(measurement['S'],measurement['R'])
    if shots is None:
        return phase,1e-12
    # 两个 +/-1 均值的联合 Hoeffding 界，单次均值误差 <= eta。
    eta=math.sqrt(2*math.log(4/per_pair_failure)/shots)
    radius=math.sqrt(2)*eta
    phase_error=np.pi/2 if radius>=1 else .5*math.asin(radius)
    return phase,float(phase_error)


def merge_intervals(intervals: Sequence[tuple[float,float]]) -> list[tuple[float,float]]:
    result=[]
    for lo,hi in sorted(intervals):
        if result and lo<=result[-1][1]+1e-14:
            result[-1]=(result[-1][0],max(result[-1][1],hi))
        else:
            result.append((float(lo),float(hi)))
    return result


def intersect_phase_candidates(prior: Sequence[tuple[float,float]],phase: float,
                                gain: float,error: float,max_aliases: int):
    """theta = gain*coefficient + bounded error (mod pi).
    枚举所有与先验相交的分支；绝不选取最接近真实系数的分支。
    """
    intervals=[]
    count=0
    for lo,hi in prior:
        first=math.ceil((gain*lo-error-phase)/np.pi-1e-12)
        last=math.floor((gain*hi+error-phase)/np.pi+1e-12)
        count+=max(0,last-first+1)
        if count>max_aliases:
            raise RuntimeError('相位别名过多；需要更紧先验或更短起始时间。')
        for k in range(first,last+1):
            left=max(lo,(phase+k*np.pi-error)/gain)
            right=min(hi,(phase+k*np.pi+error)/gain)
            if left<=right:
                intervals.append((left,right))
    return merge_intervals(intervals)


def interval_estimate(intervals):
    # 保留所有候选，半径覆盖整个候选集合；绝不隐藏未消除的别名。
    lower,upper=intervals[0][0],intervals[-1][1]
    estimate=.5*(lower+upper)
    return estimate,.5*(upper-lower)


def refine_one_coefficient(label: str, kind: str, p: Mapping, cfg: Config,
                            periods: Sequence[int], per_pair_failure: float,
                            rng: np.random.Generator) -> dict:
    center,halfwidth=p[kind+'_hat'],p['epsilon_'+kind]
    intervals=[(center-halfwidth,center+halfwidth)]
    goal=cfg.target_B_error if kind=='B' else cfg.target_C_error
    history=[]
    status='target_reached' if halfwidth<=goal else 'time_or_frequency_limit'
    for m in periods:
        if interval_estimate(intervals)[1]<=goal:
            status='target_reached'
            break
        t=2*np.pi*m/p['omega_hat']
        measured=simulate_ramsey_experiment(label,kind,m,p['omega_hat'],cfg.shots,rng)
        phase,measurement_error=phase_from_measurement(measured,cfg.shots,per_pair_failure)
        model_error=systematic_phase_bound(kind,t,p)
        gain=t if kind=='B' else 2*t/np.pi
        updated=intersect_phase_candidates(intervals,phase,gain,
            model_error+measurement_error+cfg.extra_phase_error,cfg.max_alias_candidates)
        if not updated:
            raise RuntimeError(f'{label}/{kind}: 第 m={m} 周期的区间交集为空。'
                               '检查上游区间、噪声、控制误差或模型，不能强行输出高精度结果。')
        intervals=updated
        estimate,radius=interval_estimate(intervals)
        history.append(dict(m=m,t=t,theta_mod=phase,measurement=measured,
                            measurement_phase_error=measurement_error,
                            model_phase_error_bound=model_error,
                            estimate=estimate,error_bound=radius,
                            intervals=intervals.copy()))
    estimate,radius=interval_estimate(intervals)
    if radius<=goal:
        status='target_reached'
    elif not periods:
        status='no_admissible_period'
    elif len(intervals)>1:
        status='unresolved_phase_aliases'
    return dict(estimate=estimate,error_bound=radius,intervals=intervals,
                status=status,history=history,target_error=goal,
                max_interrogation_time=max((x['t'] for x in history),default=0.),
                sum_interrogation_times=sum(x['t'] for x in history),
                # R、S 必须使用独立制备；无限 shots 的总感知时间不定义为有限值。
                total_sensing_time=None if cfg.shots is None else
                    2*cfg.shots*sum(x['t'] for x in history))


def refine_all_targets(frequency_results: Optional[Mapping]=None,
                       config: Optional[Config]=None) -> dict:
    cfg=Config() if config is None else config
    if cfg.shots is not None and (not isinstance(cfg.shots,(int,np.integer)) or cfg.shots<=0):
        raise ValueError('shots 必须为正整数或 None。')
    if not 0<cfg.measurement_failure_probability<1:
        raise ValueError('测量失败概率必须在 (0,1)。')
    if min(cfg.target_B_error,cfg.target_C_error,cfg.max_B_time,cfg.max_C_time,
           cfg.max_C_model_phase_error)<=0 or cfg.extra_phase_error<0:
        raise ValueError('目标误差/时间/模型预算必须为正，额外相位误差非负。')
    inputs=DEFAULT_INPUTS if frequency_results is None else frequency_results
    params={lab:normalize_input(rec) for lab,rec in inputs.items()}
    schedules={(lab,kind):synchronized_schedule(kind,p,cfg)
               for lab,p in params.items() for kind in ('B','C')}
    # 预先为最多可能执行的全部测量分配失败概率，允许提前停止。
    max_pairs=sum(len(x) for x in schedules.values())
    per_pair_failure=cfg.measurement_failure_probability/max(1,max_pairs)
    results={}
    for index,(label,p) in enumerate(params.items()):
        rng=np.random.default_rng(cfg.seed+1000*index)
        b=refine_one_coefficient(label,'B',p,cfg,schedules[label,'B'],per_pair_failure,rng)
        c=refine_one_coefficient(label,'C',p,cfg,schedules[label,'C'],per_pair_failure,rng)
        result=dict(target_label=label,inputs=p,B=b,C=c,
                    B_hat_ref=b['estimate'],C_hat_ref=c['estimate'],
                    epsilon_B_ref=b['error_bound'],epsilon_C_ref=c['error_bound'])
        # 事后仿真诊断，不能用于控制流程或挑选结果。
        if label in TERM_BY_LABEL:
            truth=TERM_BY_LABEL[label]
            result['diagnostics']=dict(
                B_true=truth.B,C_true=truth.C,omega_true=truth.omega,
                B_error_before=abs(p['B_hat']-truth.B),
                C_error_before=abs(p['C_hat']-truth.C),
                B_error_after=abs(b['estimate']-truth.B),
                C_error_after=abs(c['estimate']-truth.C),
                initial_intervals_valid=(abs(p['B_hat']-truth.B)<=p['epsilon_B'] and
                    abs(p['C_hat']-truth.C)<=p['epsilon_C'] and
                    p['omega_lower']<=truth.omega<=p['omega_upper']),
                final_intervals_cover_truth=(any(lo<=truth.B<=hi for lo,hi in b['intervals'])
                    and any(lo<=truth.C<=hi for lo,hi in c['intervals'])))
        results[label]=result
    return results


def print_results(results: Mapping, show_history: bool=False):
    print('\nB、C 精化（先验区间与测量/模型误差共同决定最终区间）')
    print('P_s       B_hat_ref          B_bound          C_hat_ref          C_bound')
    for lab,r in results.items():
        print(f'{lab:4s} {r["B_hat_ref"]:18.12f} {r["epsilon_B_ref"]:16.6e} '
              f'{r["C_hat_ref"]:18.12f} {r["epsilon_C_ref"]:16.6e}')
    print('\n仿真真实误差（仅用于事后比较，不是算法输入）')
    print('P_s       B_before        B_after         C_before        C_after')
    for lab,r in results.items():
        d=r.get('diagnostics')
        if d:
            print(f'{lab:4s} {d["B_error_before"]:15.6e} {d["B_error_after"]:15.6e} '
                  f'{d["C_error_before"]:15.6e} {d["C_error_after"]:15.6e}')
    translations={'target_reached':'达到目标区间精度',
                  'time_or_frequency_limit':'受时间/频率误差/测量限制',
                  'no_admissible_period':'当前预算不允许完整周期，保留初始区间',
                  'unresolved_phase_aliases':'仍存在多个相位候选区间'}
    print('\n停止状态及最长单次询问时间')
    for lab,r in results.items():
        print(f'{lab}: B: {translations[r["B"]["status"]]} '
              f'(T={r["B"]["max_interrogation_time"]:.6f}); '
              f'C: {translations[r["C"]["status"]]} '
              f'(T={r["C"]["max_interrogation_time"]:.6f})')
        if show_history:
            for kind in ('B','C'):
                for h in r[kind]['history']:
                    print(f'  {kind} m={h["m"]:4d}, t={h["t"]:.6f}, '
                          f'estimate={h["estimate"]:.12f}, radius={h["error_bound"]:.3e}, '
                          f'branches={len(h["intervals"])}')



if __name__=='__main__':
    config=Config(
        target_B_error=1e-4,
        target_C_error=1e-4,
        max_B_time=1024.,
        max_C_time=64.,
        shots=None,  
    )
    results=refine_all_targets(config=config)
    print_results(results,show_history=False)
