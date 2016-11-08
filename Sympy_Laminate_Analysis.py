import sympy as sp
import numpy as np
import math

__author__ = 'MNR'

__all__ = ["T_Matrix", "R_Matrix", "Poissons_Ratio", "get_Elastic_Constants",
           "Material", "Iso_Material", "Ortho_Material", "Ply", "GSCS_Ply",
           "Ortho_Ply", "Laminate"]


def T_Matrix(angle):
    """
    Calculates the transformation matrix.
    Parameters
    ----------
    angle : 'Int', 'Float', or Variable
        Angle in degress of desired transformation.
    """
    if isinstance(angle, (int, float)):
        theta = math.radians(angle)
    else:
        theta = angle

    c, s = (sp.cos(theta), sp.sin(theta))
    return sp.Matrix([[c ** 2, s ** 2, 2 * s * c],
                     [s ** 2, c ** 2, -2 * s * c],
                     [-s * c, s * c, c ** 2 - s ** 2]])


def R_Matrix():
    """
    Returns R matrices for transformation of S and Q in engineering
    strain_type.
    """
    return sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 2]])


def Poissons_Ratio(vij, Ei, Ej):
    """
    Calculates other poissons ratio vji = vij*(Ej/Ei)
    """
    return vij * Ej / Ei


def get_Elastic_Constants(S_Matrix, strain_type="Engineering"):
    """
    Extracts elastic constants from stiffness matrix.
    """
    if strain_type.lower().startswith('e'):
        strain_multiplier = 1
    else:
        strain_multiplier = 2

    E11 = 1 / S_Matrix[0, 0]
    E22 = 1 / S_Matrix[1, 1]
    G12 = 1 / (strain_multiplier * S_Matrix[2, 2])
    v12 = -1 * S_Matrix[1, 0] * E11
    v21 = -1 * S_Matrix[0, 1] * E22

    return E11, E22, G12, v12, v21


class Material(object):
    def __init__(self, E1, E2, G1, G2, v12, v21):
        """
        Create Material instance with given elastic constants
        Parameters
        ----------
        E1 : 'Int' or 'Float'
            Axial Young's Modulus.
        E2 : 'Int' or 'Float'
            Transverse Young's Modulus.
        G1 : 'Int' or 'Float'
            Axial Shear Modulus.
        G1 : 'Int' or 'Float'
            Transverse Shear Modulus.
        v12 : 'Int' or 'Float'
            Axial Poisson's Ratio.
        v21 : 'Int' or 'Float'
            Transverse Poisson's Ratio.
        """
        self.E1 = E1
        self.E2 = E2
        self.G1 = G1
        self.G2 = G2
        self.v12 = v12
        self.v21 = v21


class Iso_Material(Material):
    def __init__(self, E, G=None, v=None):
        """
        Calculates elastic constants for an isotropic material and passes them
        to Material.
        Parameters
        ----------
        E : 'Int' or 'Float'
            Young's Modulus.
        G : 'Int' or 'Float'
            Shear Modulus, G = E/(2(1 + v).
        v : 'Int' or 'Float'
            Poisson's Ratio, v = E/2*G - 1.
        """
        assert v is not None or G is not None, "Must supply v and/or G"

        E1 = E2 = E
        if v is None:
            v12 = v21 = E / (2 * G) - 1
        else:
            v12 = v21 = v
        if G is None:
            G1 = G2 = E / (2 * (1 + v))
        else:
            G1 = G2 = G

        # call parent constructor with simplified arguments
        Material.__init__(self, E1, E2, G1, G2, v12, v21)


class Ortho_Material(Material):
    def __init__(self, E1, E2, G1, G2, v12=None, v21=None):
        """
        Calculates elastic constants for transversely orthotropic material and
        passes them to Material.
        Parameters
        ----------
        E1 : 'Int' or 'Float'
            Axial Young's Modulus.
        E2 : 'Int' or 'Float'
            Transverse Young's Modulus.
        G1 : 'Int' or 'Float'
            Axial Shear Modulus.
        G2 : 'Int' or 'Float'
            Transverse Shear Modulus .
        v12 : 'Int' or 'Float'
            Axial Poisson's Ratio, v12 = v21(E1/E2).
        v21 : 'Int' or 'Float'
            Transverse Poisson's Ratio, v21 = v12(E2/E1).
        """
        assert v12 is not None or v21 is not None, \
            "Must supply v12 and/or v21."

        if v12 is None:
            v12 = Poissons_Ratio(v21, E2, E1)
        if v21 is None:
            v21 = Poissons_Ratio(v12, E1, E2)

        # call parent constructor with simplified arguments
        Material.__init__(self, E1, E2, G1, G2, v12, v21)


class Ply(object):
    def __init__(self, E11, E22, G12, v12, v21):
        """
        Create Ply instancd with given elastic constants
        Parameters
        ----------
        E11 : 'Int' or 'Float'
            Axial Young's Modulus.
        E22 : 'Int' or 'Float'
            Transverse Young's Modulus.
        G12 : 'Int' or 'Float'
            Shear Modulus.
        v12 : 'Int' or 'Float'
            Axial Poisson's Ratio.
        v21 : 'Int' or 'Float'
            Transverse Poisson's Ratio.
        """
        self.E11 = E11
        self.E22 = E22
        self.G12 = G12
        self.v12 = v12
        self.v21 = v21

    def get_S(self, theta=0., strain_type="Engineering"):
        """
        Calculates the compliance matrix for the ply oriented at angle theta
        in the specified strain units
        Parameters
        ----------
        theta : 'Int' or 'Float'
            Angle in degrees of ply orientation.
        strain_type : 'String'
            Specifies 'Engineering' or 'Tensorial' strain.
        """

        if strain_type.lower().startswith('e'):
            strain_multiplier = 1
        else:
            strain_multiplier = 2

        compliance = sp.Matrix([[1 / self.E11, -self.v21 / self.E22, 0],
                               [-self.v12 / self.E11, 1 / self.E22, 0],
                               [0, 0, 1 / (strain_multiplier * self.G12)]])

        if theta == 0.:
            return compliance
        else:
            T = T_Matrix(theta)
            if strain_type.lower().startswith('e'):
                R = R_Matrix()
                TI = sp.simplify(R * T.inv() * R.inv())
            else:
                TI = T.inv()
            return sp.simplify(sp.N(TI * compliance * T, chop=1e-10))

    def get_Q(self, theta=0., strain_type="Engineering"):
        """
        Calculates the stiffness matrix (Q = S^-1) for the ply oriented at
        angle theta in the specified strain units
        Parameters
        ----------
        theta : 'Int' or 'Float'
            Angle in degrees of ply orientation.
        strain_type : 'String'
            Specifies 'Engineering' or 'Tensorial' strain.
        """

        return sp.simplify(sp.N(self.get_S(theta, strain_type).inv(),
                           chop=1e-10))

    def weave_Ply(self, orientation=(0, 90)):
        """
        Calculates the elastic constants of a woven ply in the given
        orientation
        Parameters
        ----------
        orientation : 'Tuple', 'len(orientation) == 2'
            Orientation of the weave.
        """
        assert len(orientation) == 2, "orientation should have 2 entries"
        Q_weave = (self.get_Q(orientation[0]) + self.get_Q(orientation[1])) / 2
        S_weave = sp.simplify(sp.N(Q_weave.inv(), chop=1e-10))
        (self.E11,
         self.E22,
         self.G12,
         self.v12,
         self.v21) = get_Elastic_Constants(S_weave)


class GSCS_Ply(Ply):
    def __init__(self, fiber, matrix, Vf):
        """
        Calculates the elastic constants of a ply made up of fiber and matrix
        components with fiber volume fraction Vf
        using the Christensen GSCS approach.
        Parameters
        ----------
        fiber : 'Material'
            Fiber instance of Material class.
        matrix : 'Material'
            Matrix instance of Material class.
        Vf : 'Int' or 'Float'
            Volume fraction of fiber material.
        """
        self.Vf = Vf

        # Fiber Properties
        (Eaf,
         Etf,
         Gaf,
         Gtf,
         vaf,
         vtf,
         cf) = sp.symbols('Eaf Etf Gaf Gtf vaf vtf cf')

        fiberSubs = list(zip((Eaf, Etf, Gaf, Gtf, vaf, vtf, cf),
                         (fiber.E1, fiber.E2, fiber.G1, fiber.G2, fiber.v12,
                         fiber.v21, Vf)))
        kf = (Eaf * Etf / (2 * Eaf - 4 * Etf * vaf ** 2
              - 2 * Eaf * vtf)).subs(fiberSubs)
        etaf = (3 - 4 * 1 / 2 * (1 - Gtf / kf)).subs(fiberSubs)

        # Matrix Properties
        (Eam,
         Etm,
         Gam,
         Gtm,
         vam,
         vtm,
         cm) = sp.symbols('Eam Etm Gam Gtm vam vtm cm')

        matrixSubs = list(zip((Eam, Etm, Gam, Gtm, vam, vtm, cm),
                          (matrix.E1, matrix.E2, matrix.G1, matrix.G2,
                          matrix.v12, matrix.v21, 1 - Vf)))
        km = (Eam * Etm / (2 * Eam - 4 * Etm * vam ** 2
              - 2 * Eam * vtm)).subs(matrixSubs)
        # mm = (1 + 4 * km * vam ** 2 / Eam).subs(matrixSubs)
        etam = (3 - 4 * 1 / 2 * (1 - Gtm / km)).subs(matrixSubs)

        # Axial Ply Properties (Hashin)
        Eac = (Eam * cm + Eaf * cf + 4 * (vaf - vam) ** 2 * cm * cf / (cm / kf
               + cf / km + 1 / Gtm)).subs(matrixSubs + fiberSubs)
        vac = (vam * cm + vaf * cf + (vaf - vam) * (1 / km - 1 / kf) * cm
               * cf / (cm / kf + cf / km
               + 1 / Gtm)).subs(matrixSubs + fiberSubs)
        Gac = (Gam * (Gam * cm + Gaf * (1 + cf)) / (Gam * (1 + cf)
               + Gaf * cm)).subs(matrixSubs + fiberSubs)
        kc = ((km * (kf + Gtm) * cm + kf * (km + Gtm) * cf) / ((kf + Gtm) * cm
              + (km + Gtm) * cf)).subs(
            matrixSubs + fiberSubs)
        Gtr = (Gtf / Gtm).subs(matrixSubs + fiberSubs)
        mc = (1 + 4 * kc * vac ** 2 / Eac)

        # Transverse Ply Properties (Hashin)
        Achr = sp.simplify((3 * cf * cm**2 * (Gtr - 1) * (Gtr + etaf)
                           + (Gtr * etam + etaf * etam - (Gtr * etam - etaf)
                           * cf ** 3) * (cf * etam * (Gtr - 1)
                           - (Gtr * etam + 1))).subs(matrixSubs + fiberSubs))
        Bchr = sp.simplify((-3 * cf * cm**2 * (Gtr - 1) * (Gtr + etaf)
                           + 1 / 2 * (etam * Gtr + (Gtr - 1) * cf + 1)
                           * ((etam - 1) * (Gtr + etaf) - 2
                           * (Gtr * etam - etaf) * cf**3) + cf / 2
                           * (etam + 1) * (Gtr - 1) * (Gtr + etaf
                           + (Gtr * etam - etaf)
                           * cf**3)).subs(matrixSubs + fiberSubs))
        Cchr = sp.simplify((3 * cf * cm ** 2 * (Gtr - 1) * (Gtr + etaf)
                           + (etam * Gtr + (Gtr - 1) * cf + 1)
                           * (Gtr + etaf + (Gtr * etam - etaf)
                           * cf ** 3)).subs(matrixSubs + fiberSubs))

        x = sp.Symbol('x')
        sols = sp.solve(Achr * x ** 2 + 2 * Bchr * x + Cchr, x)

        Gtc = sp.simplify((Gtm * sols[-1]).subs(matrixSubs))
        vtc = sp.simplify((kc - mc * Gtc) / (kc + mc * Gtc))
        Etc = sp.simplify(2 * (1 + vtc) * Gtc)

        # call parent constructor with simplified arguments
        Ply.__init__(self, Eac, Etc, Gac, vac, Poissons_Ratio(vac, Eac, Etc))


class Ortho_Ply(Ply):
    def __init__(self, E11, E22, G12, v12=None, v21=None):
        """
        Creates ply with given elastic constants
        ----------
        E11 : 'Int' or 'Float'
            Axial Young's Modulus.
        E22 : 'Int' or 'Float'
            Transverse Young's Modulus.
        G12 : 'Int' or 'Float'
            Shear Modulus.
        v12 : 'Int' or 'Float'
            Axial Poisson's Ratio, v12 = v21(E1/E2).
        v21 : 'Int' or 'Float'
            Transverse Poisson's Ratio, v21 = v12(E2/E1).
        """
        assert v12 is not None or v21 is not None, \
            "Must supply v12 and/or v21."

        if v12 is None:
            v12 = Poissons_Ratio(v21, E22, E11)
        if v21 is None:
            v21 = Poissons_Ratio(v12, E11, E22)

        # call parent constructor with simplified arguments
        Ply.__init__(self, E11, E22, G12, v12, v21)


class Laminate(object):
    def __init__(self, Plies, Layup, t_Plies=None,
                 strain_type="Engineering"):
        """
        Calculates the elastic constants and A, B, and D matrices for a
        laminate with the given set of plies.
        Parameters
        ----------
        Plies : 'List' or 'Tuple',
            List or Tuple of plies
        Layup : 'List' or 'Tuple',
            List or Tuple of angles in degrees corresponding to the
            orientation of each ply in Plies
        t_plies : 'List' or 'Tuple', default = None
            List or Tuple of ply thicknesses for each ply in Plies
        strain_type : 'String'
            Specifies 'Engineering' or 'Tensorial' strain.
        """
        if t_Plies is None:
            assert len(Plies) == len(Layup), \
                   "Must supply the same number of Plies and angles."
        else:
            assert len(Plies) == len(Layup) and len(Plies) == len(t_Plies), \
                   "Must supply the same number of Plies and angles."

        self.Plies = Plies
        self.Layup = Layup

        if t_Plies is None:
            t = 1
            t_Plies = [t/len(self.Layup), ] * len(self.Layup)
        else:
            t = sum(t_Plies)

        z = (np.hstack((np.zeros(1), np.cumsum(t_Plies))) - t/2).tolist()

        self.t = t
        self.z = z

        A = sp.zeros(3, 3)
        B = sp.zeros(3, 3)
        D = sp.zeros(3, 3)
        for k, (ply, theta) in enumerate(list(zip(self.Plies, self.Layup))):
            Ak = sp.zeros(3, 3)
            Bk = sp.zeros(3, 3)
            Dk = sp.zeros(3, 3)
            Q_Bar = ply.get_Q(theta, strain_type=strain_type)
            z_k = self.z[k]
            z_k1 = self.z[k+1]
            for i in range(3):
                for j in range(3):
                    Ak[i, j] = (z_k1 - z_k) * Q_Bar[i, j]
                    Bk[i, j] = (1 / 2) * (z_k1 ** 2 - z_k ** 2) * Q_Bar[i, j]
                    Dk[i, j] = (1 / 3) * (z_k1 ** 3 - z_k ** 3) * Q_Bar[i, j]
            A += Ak
            B += Bk
            D += Dk

        self.A = sp.simplify(sp.N(A, chop=1e-10))
        self.B = sp.simplify(sp.N(B, chop=1e-10))
        self.D = sp.simplify(sp.N(D, chop=1e-10))

        (E11,
         E22,
         G12,
         v12,
         v21) = get_Elastic_Constants(sp.simplify(sp.N(A.inv()*self.t,
                                      chop=1e-10)), strain_type=strain_type)

        self.E11 = E11
        self.E22 = E22
        self.G12 = G12
        self.v12 = v12
        self.v21 = v21
