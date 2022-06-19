mod definitions;
mod equation_parser;

use rusqlite::{Connection, Result};
use std::collections::HashMap;

use definitions::enums::Symbol;

use crate::equation_parser::*;

fn main() {
    let conn = Connection::open("data.db");
    eprintln!("{:?}", conn);

    let formula = "(Zn2(Ca(BrO4))K(Pb)2Rb)3·2(H₂O)₂·2U2"; // CaSO₄·2(H₂O)₂·2U2

    let cycles = 100000;

    use std::time::Instant;
    let now = Instant::now();
    {
        for _ in 0..cycles {
            let r = parser::parse(formula);
            /*if let Ok(parsed) = r {
                print_sorted(parsed);
            }*/
        }
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed / cycles);
}

fn print_sorted(map: HashMap<Symbol, u32>) {
    let mut v: Vec<_> = map.into_iter().collect();
    v.sort_by(|x, y| x.0.cmp(&y.0));

    eprintln!("{:?}", v);
}
