mod definitions;
mod equation_parser;

use crate::equation_parser::*;

fn main() {
    let formula = "CaSO₄·2(H₂O)₂·2U2"; // CaSO₄·2(H₂O)₂

    let r = parser::parse2(formula);
    println!("{:?}", r);
}
