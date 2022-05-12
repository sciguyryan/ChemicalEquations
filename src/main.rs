mod definitions;
mod equation_parser;

use crate::equation_parser::*;

fn main() {
    let formula = "H2O"; // CaSO₄·(H₂O)₂

    let r = parser::parse2(formula);
    println!("{:?}", r);
}
