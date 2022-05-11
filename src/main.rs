mod definitions;
mod equation_parser;

use crate::equation_parser::*;

fn main() {
    let formula = "(NH4)3PO4";

    let r = equation_parser::parser::parse(formula);
    println!("{:?}", r);
}
