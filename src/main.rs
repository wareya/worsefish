use shakmaty::{Chess, Board, Position, Square, Color, Role, Move, Piece, san::San, fen::Epd};

fn piece_value(role : Role, first_knight : &mut bool) -> f32
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
/*
fn get_piece_value_modifier(coords : (i32, i32), role : Role) -> i32
{
    let n = (coords.0 + coords.1*8) as usize;
    
    match role
    {
        Role::Pawn => [
              0,   0,   0,   0,   0,   0,  0,   0,
             98, 134,  61,  95,  68, 126, 34, -11,
             -6,   7,  26,  31,  65,  56, 25, -20,
            -14,  13,   6,  21,  23,  12, 17, -23,
            -27,  -2,  -5,  12,  17,   6, 10, -25,
            -26,  -4,  -4, -10,   3,   3, 33, -12,
            -35,  -1, -20, -23, -15,  24, 38, -22,
              0,   0,   0,   0,   0,   0,  0,   0,
        ][n],
        Role::Knight => [
            -167, -89, -34, -49,  61, -97, -15, -107,
             -73, -41,  72,  36,  23,  62,   7,  -17,
             -47,  60,  37,  65,  84, 129,  73,   44,
              -9,  17,  19,  53,  37,  69,  18,   22,
             -13,   4,  16,  13,  28,  19,  21,   -8,
             -23,  -9,  12,  10,  19,  17,  25,  -16,
             -29, -53, -12,  -3,  -1,  18, -14,  -19,
            -105, -21, -58, -33, -17, -28, -19,  -23,
        ][n],
        Role::Bishop => [
            -29,   4, -82, -37, -25, -42,   7,  -8,
            -26,  16, -18, -13,  30,  59,  18, -47,
            -16,  37,  43,  40,  35,  50,  37,  -2,
             -4,   5,  19,  50,  37,  37,   7,  -2,
             -6,  13,  13,  26,  34,  12,  10,   4,
              0,  15,  15,  15,  14,  27,  18,  10,
              4,  15,  16,   0,   7,  21,  33,   1,
            -33,  -3, -14, -21, -13, -12, -39, -21,
        ][n],
        Role::Rook => [
             32,  42,  32,  51, 63,  9,  31,  43,
             27,  32,  58,  62, 80, 67,  26,  44,
             -5,  19,  26,  36, 17, 45,  61,  16,
            -24, -11,   7,  26, 24, 35,  -8, -20,
            -36, -26, -12,  -1,  9, -7,   6, -23,
            -45, -25, -16, -17,  3,  0,  -5, -33,
            -44, -16, -20,  -9, -1, 11,  -6, -71,
            -19, -13,   1,  17, 16,  7, -37, -26,
        ][n],
        Role::Queen => [
            -28,   0,  29,  12,  59,  44,  43,  45,
            -24, -39,  -5,   1, -16,  57,  28,  54,
            -13, -17,   7,   8,  29,  56,  47,  57,
            -27, -27, -16, -16,  -1,  17,  -2,   1,
             -9, -26,  -9, -10,  -2,  -4,   3,  -3,
            -14,   2, -11,  -2,  -5,   2,  14,   5,
            -35,  -8,  11,   2,   8,  15,  -3,   1,
             -1, -18,  -9,  10, -15, -25, -31, -50,
        ][n],
        Role::King => [
            -65,  23,  16, -15, -56, -34,   2,  13,
             29,  -1, -20,  -7,  -8,  -4, -38, -29,
             -9,  24,   2, -16, -20,   6,  22, -22,
            -17, -20, -12, -27, -30, -25, -14, -36,
            -49,  -1, -27, -39, -46, -44, -33, -51,
            -14, -14, -22, -46, -44, -30, -15, -27,
              1,   7,  -8, -64, -43, -16,   9,   8,
            -15,  36,  12, -54,   8, -28,  24,  14,
        ][n],
    }
}
*/
fn get_piece_value_modifier(coords : (i32, i32), role : Role) -> f32
{
    let centered = (coords.0 as f32 - 3.5, coords.1 as f32 - 3.5);
    let tented = (1.0 - centered.0.abs() * (1.0/3.5), 1.0 - centered.1.abs() * (1.0/3.5));
    match role
    {
        Role::Pawn => ((coords.1 as f32) - 1.0) * (1.0/7.0) * 0.5 + (tented.0 * 0.5 - 0.25),
        Role::Knight =>
        {
            let mut ret = tented.0.min(tented.1) - 0.45;
            if (coords.0 == 0 || coords.0 == 7) && (coords.1 == 0 || coords.1 == 7)
            {
                ret -= 0.2;
            }
            return ret;
        }
        Role::Bishop => tented.0.min(tented.1) * 0.5 - 0.25,
        Role::Rook => tented.0 * 0.2 - 0.1,
        // try to prevent the queen from randomly developing into the middle of the board for no reason during the opening
        Role::Queen => (1.0 - tented.0.min(tented.1)) - 0.5,
        Role::King =>
        {
            let mut ret = 0.0;
            // better value in castling target positions to encourage castling
            if (coords.0 == 2 && coords.1 == 0) || (coords.0 == 6 && coords.1 == 0)
            {
                ret += 0.5;
            }
            return ret;
        }
    }
}

fn eval_inner(pos : &Chess) -> f32
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
    
    let mut eval = 0.0f32;
    
    
    let board = pos.board();
    
    for i in 0..64
    {
        let sq = Square::new(i);
        let mut first_knight_white = true;
        let mut first_knight_black = true;
        if let Some(piece) = board.piece_at(sq)
        {
            let color_f = if piece.color == Color::White { 1.0 } else { -1.0 };
            
            let mut sq_normal = sq;
            if piece.color == Color::Black
            {
                sq_normal = sq.flip_vertical()
            }
            
            let mut value = get_piece_value_modifier(sq_to_coords(sq_normal), piece.role);// as f32 * 0.01f32;
            value = match piece.color
            {
                Color::Black => -(value + piece_value(piece.role, &mut first_knight_black)),
                Color::White =>   value + piece_value(piece.role, &mut first_knight_white) ,
            };
            eval += value;
            /*
            // reduce evaluation for pieces that are being attacked
            if piece.role != Role::Pawn
            {
                let bb = board.attacks_to(sq, !piece.color, board.occupied());
                let attack_count = bb.count();
                eval -= (attack_count as f32) * 0.2 * color_f;
            }
            
            // increase evaluation for pieces+pawns that are being defended
            let bb = board.attacks_to(sq, piece.color, board.occupied());
            let attack_count = bb.count();
            if attack_count > 0
            {
                eval += 0.05 * color_f;
            }
            //eval += (attack_count as f32) * 0.02 * color_f;
            */
        }
    }
    
    // for benchmarking microoptimizations, give every evaluation a slightly different value
    eval += (pos.zobrist_hash::<Zobrist64>(EnPassantMode::Always).0 as f64 * 0.0000000000000000001) as f32;
    
    return eval;
}

fn eval(pos : &Chess) -> f32
{
    eval_inner(pos)
}

use std::collections::HashMap;
use once_cell::sync::Lazy;
use shakmaty::zobrist::ZobristHash;
use shakmaty::zobrist::Zobrist64;
use shakmaty::EnPassantMode;

static MIN_DEPTH : u16 = 6;
static MAX_DEPTH : u16 = 10;

fn move_quality_heuristic(a : &Move) -> i32
{
    let mut a_val = 0;
    
    a_val += if a.is_promotion() { 3 } else { 0 };
    a_val += if a.is_capture() && a.role() == Role::Pawn { 2 } else { 0 };
    a_val += if a.is_capture() { 2 } else { 0 };
    a_val += if a.is_castle() { 1 } else { 0 };
    a_val
}

static USE_TTABLE : bool = true;

static mut TTABLE : Lazy<HashMap<Zobrist64, (f32, u16, i8)>> = Lazy::new(|| HashMap::new() );
static mut pos_count : u64 = 0;
fn eval_with_depth(pos : &Chess, mut alpha : f32, mut beta : f32, depth : u16, color : f32) -> f32
{
    let alpha_orig = alpha;
    assert!(color == 1.0 || color == -1.0);
    
    unsafe
    {
        pos_count += 1;
        
        if USE_TTABLE
        {
            let maybe = TTABLE.get(&pos.zobrist_hash(EnPassantMode::Always));
            if let Some((score, maybe_depth, flag)) = maybe
            {
                // flag: 0 = exact, -1 : lower bound, +1 : upper bound
                if *maybe_depth >= depth
                    // ignore cached mate values because they have depth embedded in them
                    && (*score > -100000.0 && *score < 100000.0)
                {
                    match *flag
                    {
                        0 => return *score, // exact
                        -1 => alpha = alpha.max(*score),
                        1 => beta = beta.min(*score),
                        _ => panic!("invalid internal state"),
                    }
                    if alpha >= beta
                    {
                        return *score;
                    }
                }
            }
        }
    }
    
    let mut legal_moves = pos.legal_moves();
    
    if legal_moves.len() == 0 || depth == 0
    {
        return eval(&pos) * color;
    }
    
    legal_moves.sort_by(|a, b| move_quality_heuristic(b).cmp(&move_quality_heuristic(a)) );
    
    let mut best_score : f32 = -10000000000000.0;
    for move_ in legal_moves
    {
        let mut next_pos = pos.clone();
        next_pos.play_unchecked(&move_);
        let mut score = -eval_with_depth(&next_pos, -beta, -alpha, depth - 1, -color);
        
        if score > 100000.0
        {
            score -= 1.0; // prioritize short mates
        }
        
        best_score = best_score.max(score);
        alpha = alpha.max(best_score);
        if alpha >= beta
        {
            break;
        }
    }
    
    unsafe
    {
        if USE_TTABLE
        {
            let mut flag = 0;
            if best_score <= alpha_orig
            {
                flag = 1;
            }
            else if best_score >= beta
            {
                flag = -1;
            }
            TTABLE.insert(pos.zobrist_hash(EnPassantMode::Always), (best_score, depth, flag));
        }
    }
    
    best_score
}

fn find_best(pos : &Chess, depth : u16) -> Option<(Move, f32)>
{
    let ab_ext = 10000000.0;
    let color = if pos.turn() == Color::White { 1.0 } else { -1.0 };
    
    let mut legal_moves = pos.legal_moves();
    if legal_moves.len() == 0
    {
        return None;
    }
    
    let mut best_score = -10000000000000.0;
    let mut best_score_zd = best_score;
    let mut best_move = None;
    for move_ in legal_moves
    {
        let mut next_pos = pos.clone();
        next_pos.play_unchecked(&move_);
        let score = -eval_with_depth(&next_pos, -ab_ext, ab_ext, depth - 1, -color);
        if score > best_score
        {
            best_score = score;
            best_move = Some(move_);
            best_score_zd = eval(&next_pos);
        }
    }
    unsafe
    {
        if TTABLE.len() > 4000000
        {
            TTABLE.clear();
            //println!("---- clearing table");
        }
    }
    assert!(best_move.is_some());
    let best_move = best_move.unwrap();
    println!("\npicking {:?} with score {} (depth {}) (zero-depth score {})", best_move, best_score * color, depth, best_score_zd);
    Some((best_move, best_score))
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

fn print_colorizer(x : u32, y : u32, hx : u32, hy : u32, hx2 : u32, hy2 : u32)
{
    if ((x ^ y) & 1) == 1
    {
        // bright
        if x == hx && y == hy
        {
            print!("\x1b[30;106m");
        }
        else if x == hx2 && y == hy2
        {
            print!("\x1b[30;102m");
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
        else if x == hx2 && y == hy2
        {
            print!("\x1b[30;42m");
        }
        else
        {
            print!("\x1b[30;100m");
        }
    }
}

fn print_board(pos : &Chess, highlight : Option<(u32, u32)>, highlight_lo : Option<(u32, u32)>)
{
    let hx = if let Some(h) = highlight { h.0 } else { 1000 };
    let hy = if let Some(h) = highlight { h.1 } else { 1000 };
    let hx2 = if let Some(h) = highlight_lo { h.0 } else { 1000 };
    let hy2 = if let Some(h) = highlight_lo { h.1 } else { 1000 };
    println!("");
    for _y in 0..=7
    {
        let y = 7 - _y;
        for x in 0..=7
        {
            print_colorizer(x, y, hx, hy, hx2, hy2);
            
            let piece = pos.board().piece_at(Square::new(x + y * 8));
            print!(" {} ", piece_letter(piece));
        }
        println!("\x1b[0;0m");
    }
}

fn get_desired_depth(pos : &Chess) -> u16
{
    let mut depth = MIN_DEPTH;
    let removed_pieces = 32 - pos.board().occupied().count();
    // For each 8 pieces that have been removed from the board, add 1 to depth.
    depth += (removed_pieces / 8) as u16;
    depth = depth.min(MAX_DEPTH);
    let white_pieces = pos.board().by_color(Color::White).count();
    let black_pieces = pos.board().by_color(Color::Black).count();
    // for each 4 pieces/pawns advantage one side has over the other, add 1 to depth.
    depth += ((white_pieces as isize - black_pieces as isize).abs() / 4) as u16;
    depth
}
    
fn main()
{
    // first, print out value tables for the sake of sanity
    for role in &[Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen, Role::King]
    {
        println!("Value offset grid for {:?}...", role);
        for _y in 0..=7
        {
            let y = 7 - _y;
            for x in 0..=7
            {
                print_colorizer(x, y, 1000, 1000, 1000, 1000);
                print!("{:5.2} ", get_piece_value_modifier((x as i32, y as i32), *role));
            }
            println!("\x1b[0;0m");
        }
    }
    
    let mut pos = if true
    {
        Chess::default()
    }
    else
    {
        use shakmaty::fen::Fen;
        use shakmaty::CastlingMode;
        
        //let fen = "r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6"; // four knights opening
        //let fen = "rnbqkb1r/pppppp1p/5np1/8/8/5NP1/PPPPPP1P/RNBQKB1R w KQkq - 0 3"; // symmetrical king's indian
        
        //let fen = "r1b2nk1/1pp1r3/p1p4p/4Np1N/8/3P4/PPP2P1P/R3R1K1 w - - 2 23"; // white advantage midgame
        //let fen = "1R6/2p2pk1/3p4/8/3P2r1/1P6/3r4/1K5R w - - 0 34"; // white: don't connet the rooks!
        //let fen = "5k2/2p2p2/3p4/3P4/8/1P6/1K4r1/R7 w - - 11 43"; // white: loop trigger
        //let fen = "5r2/p4pkp/8/5QPP/3P2P1/8/4q3/5RK1 w - - 1 41"; // white: don't trade down on the rook!
        
        let fen = "1r1qr1k1/pppb1ppp/2n2n2/4p3/P1Np4/1P2PN2/2PP1PPP/1R1QKB1R w K - 0 13"; // white: don't trade the knight!
        
        //let fen = "8/2p3k1/8/1P2p3/p1P5/8/P2P2PR/2K5 w - - 1 34"; // mate in 11 for white
        //let fen = "8/1P6/8/6k1/4R3/p2P4/P5P1/2K5 w - - 1 42"; // mate in 5 for white
        //let fen = "2k5/2P5/8/3b4/8/8/3K4/6q1 b - - 7 93"; // mate in 5 for black (9 ply)
        //let fen = "R7/8/8/8/8/2k4p/2P1K2P/1QR1R3 w - - 1 35"; // mate in 2 for white
        //let fen = "rnbqkb1r/p1ppp1pp/5n2/1p6/5P1P/PPN5/2PPp3/R1BQKB1R w KQkq - 0 8"; // trivially best to recapture
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
    
    print_board(&pos, None, None);
    
    let mut depth = MIN_DEPTH + 1;
    
    let mut move_limit = 8;
    
    let mut move_ = None;
    let mut n = 0;
    let mut movelog = "".to_string();
    let mut movelog_uci = "".to_string();
    while true
    {
        if pos.is_insufficient_material() || pos.halfmoves() >= 100
        {
            break;
        }
        
        depth = get_desired_depth(&pos);
        /*
        depth -= 1;
        if depth < 1
        {
            depth = 1
        }
        */
        
        let start_pos_count = unsafe { pos_count };
        move_ = find_best(&pos, depth);
        let end_pos_count = unsafe { pos_count };
        println!("positions evaluated: {}", end_pos_count);
        
        if move_.is_none()
        {
            break;
        }
        
        let m = &move_.as_ref().unwrap().0;
        let uci = m.to_uci(pos.castles().mode());
        print!("{} ", uci);
        
        use std::io::Write;
        std::io::stdout().flush().unwrap();
        
        movelog += &format!("{} ", San::from_move(&pos, &m).to_string());
        movelog_uci += &format!("{} ", uci);
        
        pos.play_unchecked(m);
        if n % 1 == 0
        {
            print_board(&pos,
                Some((m.to().file().into(), m.to().rank().into())),
                Some((m.from().map(|x| x.file().into()).unwrap_or(1000), m.from().map(|x| x.rank().into()).unwrap_or(1000))));
        }
        n += 1;
        
        if n % 16 == 0
        {
            movelog += "\n";
            movelog_uci += "\n";
        }
        
        std::io::stdout().flush().unwrap();
        
        if false
        {
            depth -= 1;
            if depth == 0
            {
                break;
            }
        }
        
        move_limit -= 1;
        if move_limit == 0
        {
            break;
        }
    }
    
    println!("");
    
    
    println!("");
    println!("All done!");
    println!("{}", movelog);
    println!("{}", movelog_uci);
    println!("{}", Epd::from_position(pos.clone(), EnPassantMode::Legal));
    print_board(&pos, None, None);
    
    let end_pos_count = unsafe { pos_count };
    println!("positions evaluated: {}", end_pos_count);
    
    if pos.is_checkmate()
    {
        println!("MATE.");
    }
}
