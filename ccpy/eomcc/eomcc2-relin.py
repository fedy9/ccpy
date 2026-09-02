import numpy as np
from ccpy.lib.core import cc_loops2

def _keep_only_active(X, o_act_idx, v_act_idx):
    """Zero every element of X=(v,v,o,o) outside the fully-active corner
    [:v_act_idx+1, :v_act_idx+1, o_act_idx:, o_act_idx:], i.e., zero the full
    complement of that corner -- not just the fully-inactive corner. Any element
    with at least one inactive index (a "mixed" active/inactive combination) must
    be excluded here, since build_hr_1A/1B folds exactly that same complement;
    leaving mixed elements nonzero here would double-count them."""
    mask = np.ones(X.shape, dtype=bool)
    mask[:v_act_idx+1, :v_act_idx+1, o_act_idx:, o_act_idx:] = False
    X[mask] = 0.0

def update(R, omega, H, fock, RHF_symmetry, system, omega_fixed, o_act_idx, v_act_idx):

    R.a, R.b, R.aa, R.ab, R.bb = cc_loops2.update_r(
        R.a,
        R.b,
        R.aa,
        R.ab,
        R.bb,
        omega,
        fock.a.oo,
        fock.a.vv,
        fock.b.oo,
        fock.b.vv,
        0.0,
    )
    _keep_only_active(R.aa, o_act_idx, v_act_idx)
    _keep_only_active(R.ab, o_act_idx, v_act_idx)
    _keep_only_active(R.bb, o_act_idx, v_act_idx)
    if RHF_symmetry:
        R.b = R.a.copy()
        R.bb = R.aa.copy()
    return R

def HR(dR, R, T, H, fock, flag_RHF, system, omega_fixed, o_act_idx, v_act_idx):

    # make sure inactive elements of input vector are set to 0
    _keep_only_active(R.aa, o_act_idx, v_act_idx)
    _keep_only_active(R.ab, o_act_idx, v_act_idx)
    _keep_only_active(R.bb, o_act_idx, v_act_idx)

    # update R1
    dR.a = build_HR_1A(R, H)
    if flag_RHF:
        dR.b = dR.a.copy()
    else:
        dR.b = build_HR_1B(R, H)
    # update R2
    dR.aa = build_HR_2A(R, T, H, fock)
    dR.ab = build_HR_2B(R, T, H, fock)
    if flag_RHF:
        dR.bb = dR.aa.copy()
    else:
        dR.bb = build_HR_2C(R, T, H, fock)

    # add R1 from inactive R2 (r2)
    dR.a += build_hr_1A(dR, H, fock, omega_fixed, o_act_idx, v_act_idx)
    if flag_RHF:
        dR.b = dR.a.copy()
    else:
        dR.b += build_hr_1B(dR, H, fock, omega_fixed, o_act_idx, v_act_idx)
    # set inactive elements of output vector to 0
    _keep_only_active(dR.aa, o_act_idx, v_act_idx)
    _keep_only_active(dR.ab, o_act_idx, v_act_idx)
    _keep_only_active(dR.bb, o_act_idx, v_act_idx)

    return dR.flatten()

def build_HR_1A(R, H):
    # < ia | [H(2)*(R1+R2)]_C | 0 >
    X1A = -np.einsum("mi,am->ai", H.a.oo, R.a, optimize=True)
    X1A += np.einsum("ae,ei->ai", H.a.vv, R.a, optimize=True)
    X1A += np.einsum("amie,em->ai", H.aa.voov, R.a, optimize=True)
    X1A += np.einsum("amie,em->ai", H.ab.voov, R.b, optimize=True)
    X1A -= 0.5 * np.einsum("mnif,afmn->ai", H.aa.ooov, R.aa, optimize=True)
    X1A -= np.einsum("mnif,afmn->ai", H.ab.ooov, R.ab, optimize=True)
    X1A += 0.5 * np.einsum("anef,efin->ai", H.aa.vovv, R.aa, optimize=True)
    X1A += np.einsum("anef,efin->ai", H.ab.vovv, R.ab, optimize=True)
    X1A += np.einsum("me,aeim->ai", H.a.ov, R.aa, optimize=True)
    X1A += np.einsum("me,aeim->ai", H.b.ov, R.ab, optimize=True)
    return X1A

def build_HR_1B(R, H):
    # < i~a~ | [H(2)*(R1+R2)]_C | 0 >
    X1B = -np.einsum("mi,am->ai", H.b.oo, R.b, optimize=True)
    X1B += np.einsum("ae,ei->ai", H.b.vv, R.b, optimize=True)
    X1B += np.einsum("maei,em->ai", H.ab.ovvo, R.a, optimize=True)
    X1B += np.einsum("amie,em->ai", H.bb.voov, R.b, optimize=True)
    X1B -= np.einsum("nmfi,fanm->ai", H.ab.oovo, R.ab, optimize=True)
    X1B -= 0.5 * np.einsum("mnif,afmn->ai", H.bb.ooov, R.bb, optimize=True)
    X1B += np.einsum("nafe,feni->ai", H.ab.ovvv, R.ab, optimize=True)
    X1B += 0.5 * np.einsum("anef,efin->ai", H.bb.vovv, R.bb, optimize=True)
    X1B += np.einsum("me,eami->ai", H.a.ov, R.ab, optimize=True)
    X1B += np.einsum("me,aeim->ai", H.b.ov, R.bb, optimize=True)
    return X1B

def build_hr_1A(dR, H, fock, omega_fixed, o_act_idx, v_act_idx):
    # < ia | [H(2)*(r2)]_C | 0 >
    # NOTE: dR.aa/ab/bb at this point holds the full residual computed from the *active*
    # R2 amplitudes alone (the inactive block of R2 was zeroed before this residual was
    # built). Restricting that residual to the inactive index block and dividing it by
    # the (omega_fixed - D2) denominator is only valid because the doubles-doubles
    # self-coupling (built with the bare, diagonal-by-construction Fock operator in
    # build_HR_2A/2B/2C) cannot leak active-block amplitudes into inactive-block indices:
    # a diagonal operator maps each index combination only to itself. This is why the
    # bare (undressed) Fock -- not the dressed CC2 Hbar -- must be used everywhere in
    # this module: only the bare Fock is truly diagonal, so the "residual restricted to
    # the inactive block" below equals exactly the R1-driven numerator restricted to
    # that block, i.e., the correct partitioned R2 numerator for the inactive amplitudes.
    # Copy tensors
    _ra = np.zeros_like(dR.a)
    _rb = np.zeros_like(dR.b)
    raa = dR.aa.copy()
    rab = dR.ab.copy()
    rbb = dR.bb.copy()
    # Set active parts to 0
    raa[:v_act_idx+1, :v_act_idx+1, o_act_idx:, o_act_idx:] = 0.0
    rab[:v_act_idx+1, :v_act_idx+1, o_act_idx:, o_act_idx:] = 0.0
    rbb[:v_act_idx+1, :v_act_idx+1, o_act_idx:, o_act_idx:] = 0.0
    # Divide by diagonal
    _ra, _rb, raa, rab, rbb = cc_loops2.update_r(
        _ra,
        _rb,
        raa,
        rab,
        rbb,
        omega_fixed,
        fock.a.oo,
        fock.a.vv,
        fock.b.oo,
        fock.b.vv,
        0.0,
    )
    # Contract into R1
    X1A = -0.5 * np.einsum("mnif,afmn->ai", H.aa.ooov, raa, optimize=True)
    X1A -= np.einsum("mnif,afmn->ai", H.ab.ooov, rab, optimize=True)
    X1A += 0.5 * np.einsum("anef,efin->ai", H.aa.vovv, raa, optimize=True)
    X1A += np.einsum("anef,efin->ai", H.ab.vovv, rab, optimize=True)
    X1A += np.einsum("me,aeim->ai", H.a.ov, raa, optimize=True)
    X1A += np.einsum("me,aeim->ai", H.b.ov, rab, optimize=True)
    return X1A

def build_hr_1B(dR, H, fock, omega_fixed, o_act_idx, v_act_idx):
    # < i~a~ | [H(2)*(r2)]_C | 0 >
    # See the note in build_hr_1A above -- the same bare-Fock-diagonal argument applies.
    # Copy tensors
    _ra = np.zeros_like(dR.a)
    _rb = np.zeros_like(dR.b)
    raa = dR.aa.copy()
    rab = dR.ab.copy()
    rbb = dR.bb.copy()
    # Set active parts to 0
    raa[:v_act_idx+1, :v_act_idx+1, o_act_idx:, o_act_idx:] = 0.0
    rab[:v_act_idx+1, :v_act_idx+1, o_act_idx:, o_act_idx:] = 0.0
    rbb[:v_act_idx+1, :v_act_idx+1, o_act_idx:, o_act_idx:] = 0.0
    # Divide by diagonal
    _ra, _rb, raa, rab, rbb = cc_loops2.update_r(
        _ra,
        _rb,
        raa,
        rab,
        rbb,
        omega_fixed,
        fock.a.oo,
        fock.a.vv,
        fock.b.oo,
        fock.b.vv,
        0.0,
    )
    # Contract into R1
    X1B = -np.einsum("nmfi,fanm->ai", H.ab.oovo, rab, optimize=True)
    X1B -= 0.5 * np.einsum("mnif,afmn->ai", H.bb.ooov, rbb, optimize=True)
    X1B += np.einsum("nafe,feni->ai", H.ab.ovvv, rab, optimize=True)
    X1B += 0.5 * np.einsum("anef,efin->ai", H.bb.vovv, rbb, optimize=True)
    X1B += np.einsum("me,eami->ai", H.a.ov, rab, optimize=True)
    X1B += np.einsum("me,aeim->ai", H.b.ov, rbb, optimize=True)
    return X1B

def build_HR_2A(R, T, H, fock):
    # < ijab | [H(2)*(R1+R2)]_C | 0 >
    # NOTE: the doubles-doubles self-coupling is diagonal by construction in CC2, i.e., it
    # is given by the bare (undressed) Fock operator, not the T1/T2-dressed CC2 Hbar.
    X2A = -0.5 * np.einsum("mi,abmj->abij", fock.a.oo, R.aa, optimize=True)  # A(ij)
    X2A += 0.5 * np.einsum("ae,ebij->abij", fock.a.vv, R.aa, optimize=True)  # A(ab)
    X2A -= 0.5 * np.einsum("bmji,am->abij", H.aa.vooo, R.a, optimize=True)  # A(ab)
    X2A += 0.5 * np.einsum("baje,ei->abij", H.aa.vvov, R.a, optimize=True)  # A(ij)
    X2A -= np.transpose(X2A, (1, 0, 2, 3)) # antisymmetrize (ab)
    X2A -= np.transpose(X2A, (0, 1, 3, 2)) # antisymmetrize (ij)
    return X2A

def build_HR_2B(R, T, H, fock):

    X2B = np.einsum("ae,ebij->abij", fock.a.vv, R.ab, optimize=True)
    X2B += np.einsum("be,aeij->abij", fock.b.vv, R.ab, optimize=True)
    X2B -= np.einsum("mi,abmj->abij", fock.a.oo, R.ab, optimize=True)
    X2B -= np.einsum("mj,abim->abij", fock.b.oo, R.ab, optimize=True)
    X2B += np.einsum("abej,ei->abij", H.ab.vvvo, R.a, optimize=True)
    X2B += np.einsum("abie,ej->abij", H.ab.vvov, R.b, optimize=True)
    X2B -= np.einsum("mbij,am->abij", H.ab.ovoo, R.a, optimize=True)
    X2B -= np.einsum("amij,bm->abij", H.ab.vooo, R.b, optimize=True)
    return X2B

def build_HR_2C(R, T, H, fock):

    X2C = -0.5 * np.einsum("mi,abmj->abij", fock.b.oo, R.bb, optimize=True)  # A(ij)
    X2C += 0.5 * np.einsum("ae,ebij->abij", fock.b.vv, R.bb, optimize=True)  # A(ab)
    X2C -= 0.5 * np.einsum("bmji,am->abij", H.bb.vooo, R.b, optimize=True)  # A(ab)
    X2C += 0.5 * np.einsum("baje,ei->abij", H.bb.vvov, R.b, optimize=True)  # A(ij)
    X2C -= np.transpose(X2C, (1, 0, 2, 3)) # antisymmetrize (ab)
    X2C -= np.transpose(X2C, (0, 1, 3, 2)) # antisymmetrize (ij)
    return X2C

