'''
Nonlinear Equation-of-Motion CC2 method.

The doubly excited amplitudes R.aa, R.ab, R.bb are never stored explicitly.
Instead, they are eliminated in favor of a partitioned expression written
entirely in terms of the singly excited amplitudes R.a, R.b, evaluated
on-the-fly at the current value of omega, and their effect is folded back
into the R1 (R.a, R.b) equations. This is exactly the same strategy used
to eliminate the triply excited amplitudes R3 in favor of R1 and R2 in
EOM-CC3 (see eomcc3.py); here, it is applied one excitation rank lower so
that only R1 remains an explicit variable.

Because R2 is fully implicit, the resulting eigenvalue problem is
nonlinear in omega (the R2 partition itself depends on omega) and must be
solved with the self-consistent, nonlinear DIIS solver used for EOM-CC3
(eomcc_nonlinear_diis), rather than the standard linear Davidson solver
used for eomcc2.py.
'''
import numpy as np
from ccpy.lib.core import eomcc_active_loops

def update(R, omega, fock, RHF_symmetry, system):
    R.a = eomcc_active_loops.update_r1a(R.a, omega, fock.a.oo, fock.a.vv, fock.b.oo, fock.b.vv, 0.0)
    if RHF_symmetry:
        R.b = R.a.copy()
    else:
        R.b = eomcc_active_loops.update_r1b(R.b, omega, fock.a.oo, fock.a.vv, fock.b.oo, fock.b.vv, 0.0)
    return R

def HR(dR, R, T, H, H1, fock, omega, flag_RHF, system):

    # Build the partitioned (implicit) doubles amplitudes R.aa, R.ab, R.bb at the
    # current omega. These are never stored on R itself; they only live for the
    # duration of this HR call, exactly as R3 is handled in eomcc3.py.
    Raa = build_R2A(R, H)
    Raa = eomcc_active_loops.update_r2a(Raa, omega, fock.a.oo, fock.a.vv, fock.b.oo, fock.b.vv, 0.0)

    Rab = build_R2B(R, H)
    Rab = eomcc_active_loops.update_r2b(Rab, omega, fock.a.oo, fock.a.vv, fock.b.oo, fock.b.vv, 0.0)

    if flag_RHF:
        Rbb = Raa.copy()
    else:
        Rbb = build_R2C(R, H)
        Rbb = eomcc_active_loops.update_r2c(Rbb, omega, fock.a.oo, fock.a.vv, fock.b.oo, fock.b.vv, 0.0)

    # Compute R1 parts of the sigma vector, folding in the implicit doubles contributions
    dR.a = build_HR_1A(R, Raa, Rab, H)
    if flag_RHF:
        dR.b = dR.a.copy()
    else:
        dR.b = build_HR_1B(R, Rab, Rbb, H)
    return dR.flatten()

def build_HR_1A(R, Raa, Rab, H):
    """< ia | [H(2)*(R1+R2)]_C | 0 >, with R2 = R2(R1; omega) evaluated implicitly"""
    X1A = -np.einsum("mi,am->ai", H.a.oo, R.a, optimize=True)
    X1A += np.einsum("ae,ei->ai", H.a.vv, R.a, optimize=True)
    X1A += np.einsum("amie,em->ai", H.aa.voov, R.a, optimize=True)
    X1A += np.einsum("amie,em->ai", H.ab.voov, R.b, optimize=True)
    X1A -= 0.5 * np.einsum("mnif,afmn->ai", H.aa.ooov, Raa, optimize=True)
    X1A -= np.einsum("mnif,afmn->ai", H.ab.ooov, Rab, optimize=True)
    X1A += 0.5 * np.einsum("anef,efin->ai", H.aa.vovv, Raa, optimize=True)
    X1A += np.einsum("anef,efin->ai", H.ab.vovv, Rab, optimize=True)
    X1A += np.einsum("me,aeim->ai", H.a.ov, Raa, optimize=True)
    X1A += np.einsum("me,aeim->ai", H.b.ov, Rab, optimize=True)
    return X1A

def build_HR_1B(R, Rab, Rbb, H):
    """< i~a~ | [H(2)*(R1+R2)]_C | 0 >, with R2 = R2(R1; omega) evaluated implicitly"""
    X1B = -np.einsum("mi,am->ai", H.b.oo, R.b, optimize=True)
    X1B += np.einsum("ae,ei->ai", H.b.vv, R.b, optimize=True)
    X1B += np.einsum("maei,em->ai", H.ab.ovvo, R.a, optimize=True)
    X1B += np.einsum("amie,em->ai", H.bb.voov, R.b, optimize=True)
    X1B -= np.einsum("nmfi,fanm->ai", H.ab.oovo, Rab, optimize=True)
    X1B -= 0.5 * np.einsum("mnif,afmn->ai", H.bb.ooov, Rbb, optimize=True)
    X1B += np.einsum("nafe,feni->ai", H.ab.ovvv, Rab, optimize=True)
    X1B += 0.5 * np.einsum("anef,efin->ai", H.bb.vovv, Rbb, optimize=True)
    X1B += np.einsum("me,eami->ai", H.a.ov, Rab, optimize=True)
    X1B += np.einsum("me,aeim->ai", H.b.ov, Rbb, optimize=True)
    return X1B

def build_R2A(R, H):
    """Numerator of the partitioned expression R.aa = X2A(R1) / (omega - D2A),
    i.e., the part of the CC2 doubles residual < ijab | [H(2)*(R1+R2)]_C | 0 >
    that does not involve R2 itself."""
    X2A = -0.5 * np.einsum("bmji,am->abij", H.aa.vooo, R.a, optimize=True)  # A(ab)
    X2A += 0.5 * np.einsum("baje,ei->abij", H.aa.vvov, R.a, optimize=True)  # A(ij)
    X2A -= np.transpose(X2A, (1, 0, 2, 3))  # antisymmetrize (ab)
    X2A -= np.transpose(X2A, (0, 1, 3, 2))  # antisymmetrize (ij)
    return X2A

def build_R2B(R, H):
    """Numerator of the partitioned expression R.ab = X2B(R1) / (omega - D2B)"""
    X2B = np.einsum("abej,ei->abij", H.ab.vvvo, R.a, optimize=True)
    X2B += np.einsum("abie,ej->abij", H.ab.vvov, R.b, optimize=True)
    X2B -= np.einsum("mbij,am->abij", H.ab.ovoo, R.a, optimize=True)
    X2B -= np.einsum("amij,bm->abij", H.ab.vooo, R.b, optimize=True)
    return X2B

def build_R2C(R, H):
    """Numerator of the partitioned expression R.bb = X2C(R1) / (omega - D2C)"""
    X2C = -0.5 * np.einsum("bmji,am->abij", H.bb.vooo, R.b, optimize=True)  # A(ab)
    X2C += 0.5 * np.einsum("baje,ei->abij", H.bb.vvov, R.b, optimize=True)  # A(ij)
    X2C -= np.transpose(X2C, (1, 0, 2, 3))  # antisymmetrize (ab)
    X2C -= np.transpose(X2C, (0, 1, 3, 2))  # antisymmetrize (ij)
    return X2C
