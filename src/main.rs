use shakmaty::{Chess, Board, Position, Square, Color, Role, Move, Piece, Rank, san::San};

fn piece_value(role : Role, first_knight : &mut bool) -> f64
{
    match role
    {
        Role::Pawn   => 1.0,
        Role::Knight =>
        {
            if *first_knight
            {
                *first_knight = false;
                return 2.5;
            }
            3.5
        }
        Role::Bishop => 3.5,
        Role::Rook   => 5.0,
        Role::Queen  => 9.0,
        Role::King   => 3.5,
    }
}

fn sq_to_coords(sq : Square) -> (i32, i32)
{
    (sq.file().into(), sq.rank().into())
}

// from white's perspective. y:0 = white piece home row, y:7 = black piece home row.
fn get_piece_value_modifier(coords : (i32, i32), role : Role) -> f64
{
    let centered = (coords.0 as f64 - 3.5, coords.1 as f64 - 3.5);
    let tented = (3.5 - centered.0.abs(), 3.5 - centered.1.abs());
    match role
    {
        Role::Pawn => ((coords.1 as f64) - 1.0) * 0.1,
        Role::Knight =>
        {
            let mut ret = tented.0.min(tented.1) * 0.4 - 0.1;
            if (coords.0 == 0 || coords.0 == 7) && (coords.1 == 0 || coords.1 == 7)
            {
                ret -= 0.2;
            }
            return ret;
        }
        Role::Bishop => tented.0.min(tented.1) * 0.3,
        Role::Rook => tented.0 * 0.1,
        Role::Queen => (3.5 - tented.0.max(tented.1)) * 0.1,
        Role::King =>
        {
            let mut ret = 0.0;
            if (coords.0 == 0 || coords.0 == 7) && (coords.1 == 0 || coords.1 == 7)
            {
                ret -= 0.8;
            }
            // better value in castling target positions to encourage castling
            if (coords.0 == 2 && coords.1 == 0) || (coords.0 == 6 || coords.1 == 0)
            {
                ret += 2.0;
            }
            return ret;
        }
    }
}

/*
use std::hash::Hash;
use std::hash::Hasher;
use std::collections::hash_map::DefaultHasher;
fn my_hash<T : Hash>(obj : T) -> u64
{
    let mut hasher = DefaultHasher::new();
    obj.hash(&mut hasher);
    hasher.finish()
}
*/

fn count_all(pos : &Chess) -> i32
{
    let mut ret = 0;
    for i in 0..64
    {
        let sq = Square::new(i);
        if let Some(piece) = pos.board().piece_at(sq)
        {
            ret += 1;
        }
    }
    ret
}

fn eval_inner(pos : &Chess) -> f64
{
    if pos.is_checkmate() && pos.turn() == Color::Black
    {
        // checkmate for white
        return 10000000.0;
    }
    if pos.is_checkmate() && pos.turn() == Color::White
    {
        // checkmate for black
        return -10000000.0;
    }
    if pos.is_stalemate() || pos.is_insufficient_material() || pos.halfmoves() > 50
    {
        return 0.0;
    }
    
    let mut eval = 0.0f64;
    
    for i in 0..64
    {
        let sq = Square::new(i);
        let mut first_knight_white = true;
        let mut first_knight_black = true;
        if let Some(piece) = pos.board().piece_at(sq)
        {
            let sq_normal = if piece.color == Color::White
            {
                sq
            }
            else
            {
                sq.flip_vertical()
            };
            
            let mut value = get_piece_value_modifier(sq_to_coords(sq_normal), piece.role);
            value = match piece.color
            {
                Color::Black => -(value + piece_value(piece.role, &mut first_knight_black)),
                Color::White =>   value + piece_value(piece.role, &mut first_knight_white) ,
            };
            eval += value;
        }
    }
    
    //eval += my_hash(pos) as f64 * 0.000000000000000000000001;
    
    return eval;
}

fn eval(pos : &Chess) -> f64
{
    eval_inner(pos)
}

fn ab_pruning(pos : &Chess, alpha : &mut f64, beta : &mut f64, score : f64) -> bool
{
    if pos.turn() == Color::White
    {
        *alpha = alpha.max(score);
        if score >= *beta
        {
            return true;
        }
    }
    else
    {
        *beta = beta.min(score);
        if score <= *alpha
        {
            return true;
        }
    }
    false
}

fn update_best(pos : &Chess, score : f64, move_ : Move, best_score : &mut f64, best_move : &mut Option<Move>)
{
    if (pos.turn() == Color::White && score > *best_score)
    || (pos.turn() == Color::Black && score < *best_score)
    {
        *best_score = score;
        *best_move = Some(move_);
    }
}

use std::collections::HashMap;
use once_cell::sync::Lazy;

static mut HASHMAP : Lazy<HashMap<Board, (Option<Move>, f64, u32)>> = Lazy::new(|| HashMap::new() );
    
static max_depth : u32 = 6;

fn find_best(pos : &Chess, mut alpha : f64, mut beta : f64, mut depth : u32) -> (Option<Move>, f64)
{
    let is_root = depth == max_depth;
    
    unsafe
    {
        let maybe = HASHMAP.get(pos.board());
        if let Some((move_, score, maybe_depth)) = maybe
        {
            // ignore cached mate values because they have depth embedded in them
            if *maybe_depth >= depth && (*score > -10000.0 && *score < 10000.0)
            {
                return (move_.clone(), *score);
            }
        }
        
        if HASHMAP.len() > 1000000
        {
            HASHMAP.clear();
        }
    }
    
    let mut best_score = if pos.turn() == Color::White { -10000000000000.0 } else { 10000000000000.0 };
    let best_score_init = best_score;
    let mut best_move = None;
    let mut legal_moves = pos.legal_moves();
    
    if legal_moves.len() == 0
    {
        let mut score = eval(&pos);
        //println!("no legal moves, returning eval... {}", score);
        return (None, score);
    }
    
    if legal_moves.len() > 1
    {
        legal_moves.sort_by(|a, b|
        {
            let get_heuristic = |a : &Move|
            {
                let mut a_val = 0;
                
                // eval function is too slow for this
                //let mut pos2 = pos.clone();
                //pos2.play_unchecked(a);
                //a_val += (eval(&pos2) * 50.0) as i32 * if pos.turn() == Color::Black { -1 } else { 1 };
                
                a_val += if a.role() == Role::Queen { 1 } else { 0 };
                a_val += if a.is_promotion() { 3 } else { 0 };
                a_val += if a.is_capture() && a.role() == Role::Pawn { 2 } else { 0 };
                a_val += if a.is_capture() { 2 } else { 0 };
                a_val += if a.is_castle() { 1 } else { 0 };
                a_val
            };
            get_heuristic(b).cmp(&get_heuristic(a))
        });
    }
    
    for move_ in legal_moves
    {
        let mut next_pos = pos.clone();
        next_pos.play_unchecked(&move_);
        let mut score;
        if depth > 0
        {
            let (next_move, _score) = find_best(&next_pos, alpha, beta, depth - 1);
            if next_move.is_none() && !next_pos.is_checkmate() && best_score_init == best_score
            {
                //println!("------------ no next move");
                continue;
            }
            score = _score;
        }
        else
        {
            score = eval(&next_pos);
        }
        
        if score < -100000.0 || score > 100000.0
        {
            score -= if (depth & 1 == 0) == (pos.turn() == Color::Black) { -1.0 } else { 1.0 };
        }
        
        update_best(pos, score, move_, &mut best_score, &mut best_move);
        
        if ab_pruning(pos, &mut alpha, &mut beta, best_score)
        {
            break;
        }
    }
    
    unsafe
    {
        HASHMAP.insert(pos.board().clone(), (best_move.clone(), best_score, depth));
    }
    
    if is_root
    {
        println!("\npicking {:?} with score {}", best_move, best_score);
    }
    
    (best_move, best_score)
}

fn piece_letter(piece : Option<Piece>) -> char
{
    let color = if let Some(piece) = piece { piece.color } else { Color :: White };
    match piece.map(|x| x.role)
    {
        None => ' ',
        Some(Role::Pawn)   => if color == Color::White { 'P' } else { 'p' },
        Some(Role::Knight) => if color == Color::White { 'N' } else { 'n' },
        Some(Role::Bishop) => if color == Color::White { 'B' } else { 'b' },
        Some(Role::Rook)   => if color == Color::White { 'R' } else { 'r' },
        Some(Role::Queen)  => if color == Color::White { 'Q' } else { 'q' },
        Some(Role::King)   => if color == Color::White { 'K' } else { 'k' },
    }
}

fn print_board(pos : &Chess, highlight : Option<(u32, u32)>)
{
    let hx = if let Some(h) = highlight { h.0 } else { 1000 };
    let hy = if let Some(h) = highlight { h.1 } else { 1000 };
    println!("");
    for _y in 0..=7
    {
        let y = 7 - _y;
        for x in 0..=7
        {
            if ((x ^ y) & 1) == 1
            {
                // bright
                if x == hx && y == hy
                {
                    print!("\x1b[30;106m");
                }
                else
                {
                    print!("\x1b[30;107m");
                }
            }
            else
            {
                // dark
                if x == hx && y == hy
                {
                    print!("\x1b[30;46m");
                }
                else
                {
                    print!("\x1b[30;100m");
                }
            }
            
            let piece = pos.board().piece_at(Square::new(x + y * 8));
            print!(" {} ", piece_letter(piece));
        }
        println!("\x1b[0;0m");
    }
}
    
fn main()
{
    let mut pos = if true
    {
        Chess::default()
    }
    else
    {
        use shakmaty::fen::Fen;
        use shakmaty::CastlingMode;
        
        let fen = "r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6"; // four knights opening
        //let fen = "8/2P2k2/PR6/4B3/3PB3/4K3/7p/2r5 w - - 2 61"; // mate in 7 for white
        //let fen = "2Q5/5k2/PR6/8/3PB3/8/r2K3B/8 w - - 1 64"; // mate in 4 for white
        //let fen = "2Q5/5k2/PR6/8/3P4/8/2BK3B/r7 w - - 3 65"; // mate in 2 for white
        //let fen = "8/2k5/P2RQ3/8/3P4/8/2BK3B/r7 w - - 11 69"; // mate in 1 for white
        //let fen = "6k1/8/6K1/8/8/8/8/4R3 w - - 41 76"; // mate in 1 for white
        //let fen = "2r3r1/5p2/5kbQ/p2p4/P2N1Bp1/5P2/1PP4P/3R1K1R w - - 4 37"; // mate in 2 for white
        //let fen = "3r3q/3k4/5q2/8/6K1/4p3/3q4/7q b - - 11 72"; // mate in 1 for black
        //let fen = "8/8/6P1/3B4/8/8/4k1K1/1Q6 w - - 11 71"; // mate in 4 for white
        //let fen = "3r3q/3k4/5q2/8/5K2/4p3/3q4/7q w - - 10 72"; // mate in 1 for black
        //let fen = "3r3q/3k4/8/8/5K2/4p3/3q4/q6q b - - 9 71"; // mate in 2 for black
        //let fen = "3r3q/3k4/8/8/4pK2/8/2p5/1q5q b - - 3 63"; // mate in 2 for black
        //let fen = "2b3k1/r1pq1ppp/np6/p2P4/N7/6PB/PPPKQ2P/RNB1R3 w - - 4 21"; // mate in 2
        //let fen = "rnbqkbnr/ppppp2p/5p2/6p1/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3"; // mate in 1
        
        fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap()
    };
    
    //eval(&pos);
    //let legals = pos.legal_moves();
    //println!("{:?}", legals);
    println!("{}", eval(&pos));
    
    let ab_ext = 10000000.0;
    
    print_board(&pos, None);
    
    let mut move_ = find_best(&pos, -ab_ext, ab_ext, max_depth);
    let mut n = 0;
    let mut movelog = "".to_string();
    while move_.0.is_some()
    {
        use std::io::Write;
        
        let m = move_.0.as_ref().unwrap();
        print!("{} ", m.to_uci(pos.castles().mode()));
        
        std::io::stdout().flush().unwrap();
        
        movelog += &format!("{} ", San::from_move(&pos, &m).to_string());
        
        pos.play_unchecked(&(move_.0.as_ref().unwrap()));
        if pos.is_game_over() || pos.halfmoves() > 50
        {
            break;
        }
        if n % 1 == 0
        {
            print_board(&pos, Some((move_.0.as_ref().unwrap().to().file().into(), move_.0.as_ref().unwrap().to().rank().into())));
        }
        n += 1;
        
        if n % 16 == 0
        {
            movelog += "\n";
        }
        
        std::io::stdout().flush().unwrap();
        
        move_ = find_best(&pos, -ab_ext, ab_ext, max_depth);
    }
    
    println!("");
    
    
    println!("");
    println!("All done!");
    println!("{}", movelog);
    print_board(&pos, None);
    if pos.is_checkmate()
    {
        println!("MATE.");
    }
}
