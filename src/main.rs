mod definitions;
mod equation_parser;

use std::collections::HashMap;

use definitions::enums::Symbol;

use crate::equation_parser::*;

fn main() {
    let formula = "(CO2)3"; // CaSO₄·2(H₂O)₂·2U2·

    let cycles = 1;

    use std::time::Instant;
    let now = Instant::now();
    {
        for i in 0..cycles {
            let r = parser::parse(formula);
            print_sorted(r);
        }
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    /*let now = Instant::now();
    {
        for i in 0..cycles {
            let r = parser::parse3(formula);
            print_sorted(r);
        }
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);*/
}

fn print_sorted(map: HashMap<Symbol, u32>) {
    let mut v: Vec<_> = map.into_iter().collect();
    v.sort_by(|x, y| x.0.cmp(&y.0));

    eprintln!("{:?}", v);
}
