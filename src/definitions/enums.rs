use std::str::FromStr;

#[allow(unused)]
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Symbol {
    H,
    He,
    Li,
    Be,
    B,
    C,
    N,
    O,
    F,
    Ne,
    Na,
    Mg,
    Al,
    Si,
    P,
    S,
    Cl,
    Ar,
    K,
    Ca,
    Sc,
    Ti,
    V,
    Cr,
    Mn,
    Fe,
    Co,
    Ni,
    Cu,
    Zn,
    Ga,
    Ge,
    As,
    Se,
    Br,
    Kr,
    Rb,
    Sr,
    Y,
    Zr,
    Nb,
    Mo,
    Tc,
    Ru,
    Rh,
    Pd,
    Ag,
    Cd,
    In,
    Sn,
    Sb,
    Te,
    I,
    Xe,
    Cs,
    Ba,
    La,
    Ce,
    Pr,
    Nd,
    Pm,
    Sm,
    Eu,
    Gd,
    Tb,
    Dy,
    Ho,
    Er,
    Tm,
    Yb,
    Lu,
    Hf,
    Ta,
    W,
    Re,
    Os,
    Ir,
    Pt,
    Au,
    Hg,
    Tl,
    Pb,
    Bi,
    Po,
    At,
    Rn,
    Fr,
    Ra,
    Ac,
    Th,
    Pa,
    U,
    Np,
    Pu,
    Am,
    Cm,
    Bk,
    Cf,
    Es,
    Fm,
    Md,
    No,
    Lr,
    Rf,
    Db,
    Sg,
    Bh,
    Hs,
    Mt,
    Ds,
    Rg,
    Cn,
    Nh,
    Fl,
    Mc,
    Lv,
    Ts,
    Og,

    // Some special case entries.
    D, // Deuterium
    T, // Tritium
}

impl FromStr for Symbol {
    type Err = ();

    fn from_str(input: &str) -> Result<Symbol, Self::Err> {
        match input {
            "H" => Ok(Symbol::H),
            "He" => Ok(Symbol::He),
            "Li" => Ok(Symbol::Li),
            "Be" => Ok(Symbol::Be),
            "B" => Ok(Symbol::B),
            "C" => Ok(Symbol::C),
            "N" => Ok(Symbol::N),
            "O" => Ok(Symbol::O),
            "F" => Ok(Symbol::F),
            "Ne" => Ok(Symbol::Ne),
            "Na" => Ok(Symbol::Na),
            "Mg" => Ok(Symbol::Mg),
            "Al" => Ok(Symbol::Al),
            "Si" => Ok(Symbol::Si),
            "P" => Ok(Symbol::P),
            "S" => Ok(Symbol::S),
            "Cl" => Ok(Symbol::Cl),
            "Ar" => Ok(Symbol::Ar),
            "K" => Ok(Symbol::K),
            "Ca" => Ok(Symbol::Ca),
            "Sc" => Ok(Symbol::Sc),
            "Ti" => Ok(Symbol::Ti),
            "V" => Ok(Symbol::V),
            "Cr" => Ok(Symbol::Cr),
            "Mn" => Ok(Symbol::Mn),
            "Fe" => Ok(Symbol::Fe),
            "Co" => Ok(Symbol::Co),
            "Ni" => Ok(Symbol::Ni),
            "Cu" => Ok(Symbol::Cu),
            "Zn" => Ok(Symbol::Zn),
            "Ga" => Ok(Symbol::Ga),
            "Ge" => Ok(Symbol::Ge),
            "As" => Ok(Symbol::As),
            "Se" => Ok(Symbol::Se),
            "Br" => Ok(Symbol::Br),
            "Kr" => Ok(Symbol::Kr),
            "Rb" => Ok(Symbol::Rb),
            "Sr" => Ok(Symbol::Sr),
            "Y" => Ok(Symbol::Y),
            "Zr" => Ok(Symbol::Zr),
            "Nb" => Ok(Symbol::Nb),
            "Mo" => Ok(Symbol::Mo),
            "Tc" => Ok(Symbol::Tc),
            "Ru" => Ok(Symbol::Ru),
            "Rh" => Ok(Symbol::Rh),
            "Pd" => Ok(Symbol::Pd),
            "Ag" => Ok(Symbol::Ag),
            "Cd" => Ok(Symbol::Cd),
            "In" => Ok(Symbol::In),
            "Sn" => Ok(Symbol::Sn),
            "Sb" => Ok(Symbol::Sb),
            "Te" => Ok(Symbol::Te),
            "I" => Ok(Symbol::I),
            "Xe" => Ok(Symbol::Xe),
            "Cs" => Ok(Symbol::Cs),
            "Ba" => Ok(Symbol::Ba),
            "La" => Ok(Symbol::La),
            "Ce" => Ok(Symbol::Ce),
            "Pr" => Ok(Symbol::Pr),
            "Nd" => Ok(Symbol::Nd),
            "Pm" => Ok(Symbol::Pm),
            "Sm" => Ok(Symbol::Sm),
            "Eu" => Ok(Symbol::Eu),
            "Gd" => Ok(Symbol::Gd),
            "Tb" => Ok(Symbol::Tb),
            "Dy" => Ok(Symbol::Dy),
            "Ho" => Ok(Symbol::Ho),
            "Er" => Ok(Symbol::Er),
            "Tm" => Ok(Symbol::Tm),
            "Yb" => Ok(Symbol::Yb),
            "Lu" => Ok(Symbol::Lu),
            "Hf" => Ok(Symbol::Hf),
            "Ta" => Ok(Symbol::Ta),
            "W" => Ok(Symbol::W),
            "Re" => Ok(Symbol::Re),
            "Os" => Ok(Symbol::Os),
            "Ir" => Ok(Symbol::Ir),
            "Pt" => Ok(Symbol::Pt),
            "Au" => Ok(Symbol::Au),
            "Hg" => Ok(Symbol::Hg),
            "Tl" => Ok(Symbol::Tl),
            "Pb" => Ok(Symbol::Pb),
            "Bi" => Ok(Symbol::Bi),
            "Po" => Ok(Symbol::Po),
            "At" => Ok(Symbol::At),
            "Rn" => Ok(Symbol::Rn),
            "Fr" => Ok(Symbol::Fr),
            "Ra" => Ok(Symbol::Ra),
            "Ac" => Ok(Symbol::Ac),
            "Th" => Ok(Symbol::Th),
            "Pa" => Ok(Symbol::Pa),
            "U" => Ok(Symbol::U),
            "Np" => Ok(Symbol::Np),
            "Pu" => Ok(Symbol::Pu),
            "Am" => Ok(Symbol::Am),
            "Cm" => Ok(Symbol::Cm),
            "Bk" => Ok(Symbol::Bk),
            "Cf" => Ok(Symbol::Cf),
            "Es" => Ok(Symbol::Es),
            "Fm" => Ok(Symbol::Fm),
            "Md" => Ok(Symbol::Md),
            "No" => Ok(Symbol::No),
            "Lr" => Ok(Symbol::Lr),
            "Rf" => Ok(Symbol::Rf),
            "Db" => Ok(Symbol::Db),
            "Sg" => Ok(Symbol::Sg),
            "Bh" => Ok(Symbol::Bh),
            "Hs" => Ok(Symbol::Hs),
            "Mt" => Ok(Symbol::Mt),
            "Ds" => Ok(Symbol::Ds),
            "Rg" => Ok(Symbol::Rg),
            "Cn" => Ok(Symbol::Cn),
            "Nh" => Ok(Symbol::Nh),
            "Fl" => Ok(Symbol::Fl),
            "Mc" => Ok(Symbol::Mc),
            "Lv" => Ok(Symbol::Lv),
            "Ts" => Ok(Symbol::Ts),
            "Og" => Ok(Symbol::Og),

            // Some special case entries.
            "D" => Ok(Symbol::D),
            "T" => Ok(Symbol::T),

            // Everything else is invalid.
            _ => Err(()),
        }
    }
}

#[allow(unused)]
enum Elements {
    Hydrogen,
    Helium,
    Lithium,
    Beryllium,
    Boron,
    Carbon,
    Nitrogen,
    Oxygen,
    Fluorine,
    Neon,
    Sodium,
    Magnesium,
    Aluminium,
    Silicon,
    Phosphorus,
    Sulfur,
    Chlorine,
    Argon,
    Potassium,
    Calcium,
    Scandium,
    Titanium,
    Vanadium,
    Chromium,
    Manganese,
    Iron,
    Cobalt,
    Nickel,
    Copper,
    Zinc,
    Gallium,
    Germanium,
    Arsenic,
    Selenium,
    Bromine,
    Krypton,
    Rubidium,
    Strontium,
    Yttrium,
    Zirconium,
    Niobium,
    Molybdenum,
    Technetium,
    Ruthenium,
    Rhodium,
    Palladium,
    Silver,
    Cadmium,
    Indium,
    Tin,
    Antimony,
    Tellurium,
    Iodine,
    Xenon,
    Caesium,
    Barium,
    Lanthanum,
    Cerium,
    Praseodymium,
    Neodymium,
    Promethium,
    Samarium,
    Europium,
    Gadolinium,
    Terbium,
    Dysprosium,
    Holmium,
    Erbium,
    Thulium,
    Ytterbium,
    Lutetium,
    Hafnium,
    Tantalum,
    Tungsten,
    Rhenium,
    Osmium,
    Iridium,
    Platinum,
    Gold,
    Mercury,
    Thallium,
    Lead,
    Bismuth,
    Polonium,
    Astatine,
    Radon,
    Francium,
    Radium,
    Actinium,
    Thorium,
    Protactinium,
    Uranium,
    Neptunium,
    Plutonium,
    Americium,
    Curium,
    Berkelium,
    Californium,
    Einsteinium,
    Fermium,
    Mendelevium,
    Nobelium,
    Lawrencium,
    Rutherfordium,
    Dubnium,
    Seaborgium,
    Bohrium,
    Hassium,
    Meitnerium,
    Darmstadtium,
    Roentgenium,
    Copernicium,
    Nihonium,
    Flerovium,
    Moscovium,
    Livermorium,
    Tennessine,
    Oganesson,

    // Some special case entries.
    Deuterium,
    Tritium,
}
