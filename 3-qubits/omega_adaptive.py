
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple
import math
import numpy as np
from scipy.optimize import brentq, least_squares

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)

from three_qubit_model import ThreeQubitOracle
DEFAULT_ORACLE=ThreeQubitOracle()

EXTRA_B_ERROR = 0.0
EXTRA_C_ERROR = 0.0

@dataclass(frozen=True)
class SearchDesign:
    B_hat: float
    C_hat: float
    epsilon_B: float
    epsilon_C: float
    C_minus: float           # 原始幅度 |C| 的区间
    C_plus: float
    scale: float
    T_total: float
    correction_bound: float
    alpha_min: float
    alpha_max: float
    a_max: float
    a_out: float
    a_in: float
    r_in: float
    r_out: float
    omega_min: float
    omega_max: float
    number_of_bins: int
    filter_epsilon: float
    failure_probability: float


def choose_search_design(coefficients: Mapping[str, float],
                         omega_min: float = 1.0, omega_max: float = 1.5,
                         failure_probability: float = .05,
                         correction_target: float = .02,
                         extra_B_error: float = EXTRA_B_ERROR,
                         extra_C_error: float = EXTRA_C_ERROR,
                         maximum_bins: int = 4096) -> SearchDesign:
    """只使用估计值、误差区间和频率先验，逐项选择四个判别参数。

    A = alpha*sinc((omega-omega_k)*T) + E,
    alpha in [pi*C_minus/(2*C_plus), pi/2], |E| <= correction_bound.
    加入余项后设计内外区阈值；旁瓣界保证外区不是只在主瓣上成立。
    """
    bh = float(coefficients.get('B_hat', coefficients.get('B_hat_fit', np.nan)))
    ch = float(coefficients['C_hat'])
    eb = float(coefficients['epsilon_B']) + extra_B_error
    ec = float(coefficients['epsilon_C']) + extra_C_error
    if not np.all(np.isfinite([bh, ch, eb, ec])) or min(eb, ec) < 0:
        raise ValueError('B/C 估计与误差界必须有限，误差半宽必须非负。')
    if not np.isfinite([omega_min,omega_max]).all() or not 0 < omega_min < omega_max < 3 * omega_min:
        raise ValueError('当前谐波余项界要求 0 < omega_min < omega_max < 3*omega_min。')
    if not 0 < failure_probability <= .5 or not 0 < correction_target <= .05:
        raise ValueError('failure_probability 应在 (0,.5]，correction_target 在 (0,.05]。')
    cminus, cplus = abs(ch) - ec, abs(ch) + ec
    if cminus <= 0 or cplus >= 2*cminus:
        raise ValueError('幅度区间不够紧：需要 |C_hat|-epsilon_C>0 且 C_plus<2*C_minus。'
                         '请提高上游 B/C 精度；不能强行截断区间。')
    gamma = 3 - omega_max / omega_min
    # 缩放 C 和残余 delta_B，使有限时间与静态项修正均按 scale 缩小。
    t_unscaled = np.pi**2 / (4*cplus)
    e_unscaled = (np.pi/(2*omega_min*t_unscaled)*(1+1/(3*gamma))
                  + np.pi*eb/(2*omega_min))
    scale = min(1.0, correction_target / e_unscaled)
    t = t_unscaled / scale
    corr = scale * e_unscaled
    amin, amax = np.pi*cminus/(2*cplus), np.pi/2
    # rho1 < .218: 第一旁瓣以外所有 sinc 旁瓣的保守统一界。
    outer_floor = amax * .218 + corr
    gap = amin - corr - outer_floor
    if gap <= 0:
        raise ValueError('余项过大，无法得到非空判别间隙。')
    aout = outer_floor + .10*gap
    ain = outer_floor + .85*gap
    sinc = lambda x: float(np.sinc(x/np.pi))
    # 内区留 10% 的半径余量，外区半径向外放宽 2%。
    rin = .90*brentq(lambda x: amin*sinc(x)-corr-ain, 1e-12, np.pi)
    rout = min(np.pi, 1.02*brentq(lambda x: amax*sinc(x)+corr-aout,
                                1e-12, np.pi))
    # 中心网格完全位于先验区间内；将 K 取二次幂便于量子寄存器初始化。
    required = max(2, math.ceil((omega_max-omega_min)*t/(2*rin)))
    n = 1 << (required-1).bit_length()
    if not isinstance(maximum_bins,int) or maximum_bins<2:
        raise ValueError('maximum_bins 必须是至少为 2 的整数。')
    if n > maximum_bins:
        raise ValueError(f'需要 {n} 个候选点，超过 maximum_bins={maximum_bins}。')
    return SearchDesign(bh, ch, eb, ec, cminus, cplus, scale, t, corr,
                        amin, amax, amax+corr, aout, ain, rin, rout,
                        omega_min, omega_max, n, failure_probability/(2*n),
                        failure_probability)


def build_frequency_grid(design: SearchDesign) -> np.ndarray:
    width = design.omega_max-design.omega_min
    step = width/design.number_of_bins
    if step*design.T_total > 2*design.r_in*(1+1e-12):
        raise RuntimeError('频率网格未覆盖内标记区。')
    return design.omega_min + (np.arange(design.number_of_bins)+.5)*step


def esu_pulse_times(omega_k: float, T_total: float) -> np.ndarray:
    if omega_k <= 0 or T_total <= 0:
        raise ValueError('频率与演化时间必须为正。')
    n = np.arange(max(0, int(np.ceil(omega_k*T_total/np.pi+.5))))
    times = (2*n+1)*np.pi/(2*omega_k)
    return times[times < T_total]


def esu_filter_integrals(omega_k: float, omega_true: float,
                         T_total: float) -> Tuple[float, float]:
    """在方波跳变点分段，精确积分；不采用主瓣近似生成模拟数据。"""
    boundaries = np.r_[0., esu_pulse_times(omega_k, T_total), T_total]
    signs = (-1.)**np.arange(len(boundaries)-1)
    g = np.dot(signs, np.diff(boundaries))
    f = np.dot(signs, np.diff(np.sin(omega_true*boundaries)))/omega_true
    return float(g), float(f)


def sensing_phases_for_target(target_label: str, omega_bins: Sequence[float],
                              design: SearchDesign) -> np.ndarray:
    """模拟未知信号源；真值仅在信号源内部使用。"""
    return np.array([DEFAULT_ORACLE.esu_phase(target_label,w,design.T_total,design.B_hat,design.scale)
                     for w in omega_bins])


def x_phase_matrix(phi: float) -> np.ndarray:
    return np.cos(phi)*I2 + 1j*np.sin(phi)*X


def build_qsp_oracle_blocks(theta_values: Sequence[float],
                            qsp_phases: Sequence[float]) -> np.ndarray:
    theta = np.asarray(theta_values, float)
    u = np.broadcast_to(x_phase_matrix(qsp_phases[0]), (len(theta),2,2)).copy()
    w = np.stack([np.exp(-1j*theta), np.exp(1j*theta)], axis=1)
    for phi in qsp_phases[1:]:
        u *= w[:,:,None]
        u = np.einsum('ij,kjl->kil', x_phase_matrix(phi), u)
    return u


def qsp_2x2_unitary(theta: float, qsp_phases: Sequence[float]) -> np.ndarray:
    return build_qsp_oracle_blocks([theta], qsp_phases)[0]


def _qsp_fit_residual_jacobian(phases: np.ndarray, theta: np.ndarray,
                               n_stop: int):
    v = np.zeros((len(theta),2), complex)
    v[:,0] = 1
    derivative = np.zeros((len(theta),2,len(phases)), complex)
    w = np.stack([np.exp(-1j*theta), np.exp(1j*theta)], axis=1)
    for k, phi in enumerate(phases):
        if k:
            v *= w
            derivative *= w[:,:,None]
        c, s = np.cos(phi), np.sin(phi)
        v = c*v + 1j*s*v[:,::-1]
        derivative = c*derivative + 1j*s*derivative[:,::-1,:]
        derivative[:,:,k] += 1j*v[:,::-1]
    # 外区压低 |1> 幅度；内区压低 |0> 幅度。
    error = np.r_[v[:n_stop,1],v[n_stop:,0]]
    jac = np.concatenate([derivative[:n_stop,1,:],derivative[n_stop:,0,:]])
    return np.r_[error.real,error.imag], np.r_[jac.real,jac.imag]


def validate_qsp_filter(phases: Sequence[float], design: SearchDesign) -> dict:
    """以酉矩阵导数范数 <= d 为连续区间补足网格验证。"""
    d = len(phases)-1
    step_max = np.sqrt(design.filter_epsilon)/(8*max(d,1))
    worst = 0.0
    for lo, hi, row in [(-design.a_out,design.a_out,1),
                        (design.a_in,design.a_max,0),
                        (-design.a_max,-design.a_in,0)]:
        n = max(2, math.ceil((hi-lo)/step_max)+1)
        theta = np.linspace(lo,hi,n)
        max_amplitude = 0.0
        for start in range(0,n,10000):
            u = build_qsp_oracle_blocks(theta[start:start+10000],phases)
            max_amplitude = max(max_amplitude,float(np.max(abs(u[:,row,0]))))
        amplitude_bound = max_amplitude+d*(hi-lo)/(2*(n-1))+1e-12
        worst = max(worst,amplitude_bound**2)
    return {'continuous_error_bound':worst,
            'filter_epsilon':design.filter_epsilon,
            'passed':worst <= design.filter_epsilon,
            'certificate_scope':'Lipschitz grid certificate in floating-point arithmetic'}

# 相同设计的已验证相位在一次运行内缓存；输入变化后自动重新合成。
QSP_PHASE_CACHE = {}


def _design_cache_key(design: SearchDesign) -> tuple:
    return (design.a_out,design.a_in,design.a_max,design.filter_epsilon)


def synthesize_qsp_phases(design: SearchDesign, seed: int = 321,
                          verbose: bool = True) -> Tuple[np.ndarray, dict]:
    key = _design_cache_key(design)
    if key in QSP_PHASE_CACHE:
        phases = np.asarray(QSP_PHASE_CACHE[key],float)
        check = validate_qsp_filter(phases,design)
        if check['passed']:
            return phases,check
    # 拟合可利用关于 theta 的对称性，验证时仍检查正、负完整区间。
    theta = np.r_[np.linspace(0,design.a_out,100),
                   np.linspace(design.a_in,design.a_max,100)]
    rng = np.random.default_rng(seed)
    for degree in (9,17,25,33,49,65,81):
        for attempt in range(2):
            last_x = last_value = None
            def evaluate(x):
                nonlocal last_x,last_value
                if last_x is None or not np.array_equal(x,last_x):
                    last_x = x.copy()
                    last_value = _qsp_fit_residual_jacobian(x,theta,100)
                return last_value
            result = least_squares(lambda p:evaluate(p)[0],
                                   rng.normal(0,.7,degree+1),
                                   jac=lambda p:evaluate(p)[1],
                                   max_nfev=450,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            residual = evaluate(result.x)[0]
            sampled_error = float(np.max(residual[:200]**2+residual[200:]**2))
            if verbose:
                print(f'  QSP degree={degree:2d}, attempt={attempt+1}, '
                      f'grid error={sampled_error:.3e}',flush=True)
            if sampled_error < design.filter_epsilon*.65:
                check = validate_qsp_filter(result.x,design)
                if check['passed']:
                    QSP_PHASE_CACHE[key] = result.x.tolist()
                    return result.x,check
    raise RuntimeError('QSP 相位合成未通过误差检查；没有执行未经验证的 BBHT。')


def apply_qsp_oracle(state: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    return np.einsum('kij,kj->ki',blocks,state)


def apply_good_reflection(state: np.ndarray) -> np.ndarray:
    out = state.copy()
    out[:,1] *= -1
    return out


def apply_initial_reflection(state: np.ndarray) -> np.ndarray:
    # I-2|uniform,0><uniform,0| = H R_0 H；不是对两个辅助分量同时扩散。
    out = state.copy()
    out[:,0] -= 2*np.mean(out[:,0])
    return out


def bbht_amplitude_search(blocks: np.ndarray, degree: int, seed: int = 42,
                          lambda_bbht: float = 1.2,
                          max_bbht_rounds: int = 150) -> dict:
    """G = -U (I-2|uniform,0><uniform,0|) U^dagger (I-2 Pi)。
    接受条件仅为辅助比特测得 1；不读取真值、不预先列举 marked_indices。
    """
    if not 1 < lambda_bbht < 4/3:
        raise ValueError('BBHT 增长因子必须满足 1 < lambda_bbht < 4/3。')
    n = len(blocks)
    if n<1 or degree<1 or max_bbht_rounds<1:
        raise ValueError('Invalid BBHT dimensions or runtime cap.')
    rng = np.random.default_rng(seed)
    dagger = blocks.conj().transpose(0,2,1)
    initial = np.zeros((n,2),complex)
    initial[:,0] = 1/np.sqrt(n)
    m = 1.0
    calls = 0
    for round_number in range(1,max_bbht_rounds+1):
        j = int(rng.integers(0,max(1,math.ceil(m))))
        state = apply_qsp_oracle(initial,blocks)
        calls += 1   # 每轮初态制备也消耗一次 QSP
        for _ in range(j):
            state = apply_good_reflection(state)
            state = apply_qsp_oracle(state,dagger)
            state = apply_initial_reflection(state)
            state = -apply_qsp_oracle(state,blocks)
            calls += 2
        probabilities = np.abs(state)**2
        norm = float(probabilities.sum())
        if abs(norm-1)>1e-8:
            raise RuntimeError(f'酉演化归一化错误: {norm}')
        sampled = int(rng.choice(2*n,p=(probabilities/norm).reshape(-1)))
        k, ancilla = divmod(sampled,2)
        if ancilla == 1:
            return {'k_hat':k,'bbht_rounds':round_number,
                    'grover_iterations_last_round':j,'total_oracle_calls':calls,
                    'esu_queries':calls*degree}
        # p >= (1-epsilon)/K；sqrt(K) 量级已足以获得常数成功概率。
        m = min(lambda_bbht*m,math.sqrt(n))
    raise RuntimeError('BBHT 达到轮数上限而未测到辅助比特 1；不返回伪结果。')


def run_qsp_bbht_frequency_circuit(target_label: str,
                                  coefficients: Mapping[str,float],
                                  omega_min: float = 1.0, omega_max: float = 1.5,
                                  failure_probability: float = .05,
                                  seed: int = 42, verbose: bool = True,
                                  sensing_callback=None, diagnostic_source=None, **design_options) -> dict:
    design = choose_search_design(coefficients,omega_min,omega_max,
                                  failure_probability,**design_options)
    bins = build_frequency_grid(design)
    if verbose:
        print(f'\n{target_label}: T={design.T_total:.6f}, scale={design.scale:.6f}, '
              f'K={len(bins)}, (r_in,r_out,a_in,a_out)='
              f'({design.r_in:.6f},{design.r_out:.6f},'
              f'{design.a_in:.6f},{design.a_out:.6f})',flush=True)
    phases, check = synthesize_qsp_phases(design,verbose=verbose)
    theta = (sensing_phases_for_target(target_label,bins,design) if sensing_callback is None
             else np.asarray(sensing_callback(bins,design),float))
    if theta.shape != bins.shape or not np.isfinite(theta).all():
        raise ValueError('Invalid ESU response shape or nonfinite phases.')
    blocks = build_qsp_oracle_blocks(theta,phases)
    search = bbht_amplitude_search(blocks,len(phases)-1,seed=seed)
    estimate = float(bins[search['k_hat']])

    # 以下诊断只能用于已知模型的仿真报告，不参与设计、搜索或接受判断。
    diagnostic_source=DEFAULT_ORACLE if diagnostic_source is None else diagnostic_source
    term = diagnostic_source.by_label[target_label]
    x = abs(bins-term.omega)*design.T_total
    inner, outer = x <= design.r_in, x >= design.r_out
    good_prob = abs(blocks[:,1,0])**2
    p_sum = float(good_prob.mean())
    conditional_bad = float(good_prob[outer].sum()/good_prob.sum()) if good_prob.sum()>0 else 1.
    interval_ok = (abs(term.B-design.B_hat) <= design.epsilon_B
                   and design.C_minus <= abs(term.C) <= design.C_plus)
    promise_ok = (np.any(inner)
                  and np.all(abs(theta[inner]) >= design.a_in-1e-10)
                  and np.all(abs(theta[outer]) <= design.a_out+1e-10)
                  and np.all(abs(theta) <= design.a_max+1e-10))
    error = abs(estimate-term.omega)
    radius = error*design.T_total
    region = 'inner' if radius<=design.r_in else ('gap' if radius<design.r_out else 'outer')
    return dict(target_label=target_label,design=design,omega_hat=estimate,
                true_omega=term.omega,absolute_error=error,
                frequency_error_bound=design.r_out/design.T_total,
                search_failure_probability_bound=failure_probability,
                upstream_failure_probability=coefficients.get('failure_probability',None),
                conditional_on='valid upstream intervals, ideal coherent ESU and filter certificate',
                backend='classical_statevector_of_ideal_coherent_ESU_QSP_BBHT',
                total_sensing_time=search['esu_queries']*design.T_total,
                returned_region=region,omega_bins=bins,theta_values=theta,
                qsp_phases=phases,qsp_degree=len(phases)-1,filter_check=check,
                p_sum=p_sum,conditional_bad_probability=conditional_bad,
                source_intervals_cover_truth=interval_ok,
                actual_response_passes_promises=bool(promise_ok),**search)


def learn_omega_for_all_targets(learned_coefficients=None,omega_min=1.,omega_max=1.5,
                                failure_probability=.05,seed=42,verbose=True,
                                source=None,**design_options):
    """failure_probability is a JOINT budget for all supplied target terms."""
    source=DEFAULT_ORACLE if source is None else source
    if learned_coefficients is None:
        from bc_estimation import learn_B_and_C_for_all_targets
        learned_coefficients=learn_B_and_C_for_all_targets(source=source)
    if not learned_coefficients:raise ValueError('No target coefficients supplied.')
    results={}
    for i,(label,coeff) in enumerate(learned_coefficients.items()):
        callback=lambda bins,d,lab=label: np.array([source.esu_phase(lab,w,d.T_total,d.B_hat,d.scale) for w in bins])
        results[label]=run_qsp_bbht_frequency_circuit(label,coeff,omega_min,omega_max,
            failure_probability/len(learned_coefficients),seed+1000*i,verbose,
            sensing_callback=callback,diagnostic_source=source,**design_options)
    return results


def print_results(results):
    print('Pauli   omega_hat       error_bound      lambda       T_ESU       QSP_degree')
    for lab,r in results.items():
        d=r['design']
        print(f'{lab:4s} {r["omega_hat"]:14.9f} {r["frequency_error_bound"]:14.6e} '
              f'{d.scale:11.6f} {d.T_total:12.6f} {r["qsp_degree"]:8d}')
        print('  filter certificate:',r['filter_check']['passed'],
              'input intervals:',r['source_intervals_cover_truth'],
              'ESU promises:',r['actual_response_passes_promises'])

if __name__=='__main__':
    print_results(learn_omega_for_all_targets())
