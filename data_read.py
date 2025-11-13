import os
import glob
import orjson
import duckdb
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
import gzip

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths - now path independent
DATA_DIR = Path(r"C:/Users/india/Downloads/Compressed/ipl_json")
DB_PATH = "IPL_cricket.db"
BATCH_SIZE = 200  # Batch size for bulk inserts
MAX_WORKERS = min(32, os.cpu_count() * 2)   #autotune based on CPU cores

@dataclass
class MatchInfo:
    """Match metadata with type safety"""
    match_id: str
    date: str
    venue: str
    city: str
    teams: List[str]
    series_name: str
    match_number: int
    toss_winner: str
    toss_decision: str
    winner: str
    margin: str
    method: str
    match_type: str
    season: str
    player_of_match: str
    original_json: bytes  # Now compressed bytes

def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Create optimized tables with indexes"""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY,
        date DATE,
        venue TEXT,
        city TEXT,
        teams TEXT[],
        series_name TEXT,
        match_number INTEGER,
        toss_winner TEXT,
        toss_decision TEXT,
        winner TEXT,
        margin TEXT,
        method TEXT,
        match_type TEXT,
        season TEXT,
        player_of_match TEXT,
        original_json BLOB  -- Changed to BLOB for compressed data
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS deliveries (
        match_id TEXT,
        inning INTEGER,
        over_num INTEGER,
        ball_num INTEGER,
        batter TEXT,
        bowler TEXT,
        non_striker TEXT,
        runs_batter INTEGER,
        runs_extras INTEGER,
        runs_total INTEGER,
        wides INTEGER,
        noballs INTEGER,
        byes INTEGER,
        legbyes INTEGER,
        wicket BOOLEAN,
        wicket_kind TEXT,
        player_out TEXT,
        fielders TEXT
    )
    """)
    logger.info(conn.execute("DESCRIBE matches").fetchall())
    logger.info(conn.execute("DESCRIBE deliveries").fetchall())
    # Create indexes for common queries
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_match_date ON matches(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_match_series ON matches(series_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_match_winner ON matches(winner)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_delivery_match ON deliveries(match_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_delivery_batter ON deliveries(batter)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_delivery_bowler ON deliveries(bowler)")
    except Exception as e:
        logger.debug(f"Index creation (may already exist): {e}")

def extract_match_info(data: Dict, filepath: str) -> MatchInfo:
    """Extract match metadata with safer field access"""
    info = data["info"]
    event = info.get("event", {})
    outcome = info.get("outcome", {})
    toss = info.get("toss", {})
    
    # Build margin string
    margin = ""
    if "by" in outcome:
        by = outcome["by"]
        if "wickets" in by:
            margin = f"{by['wickets']} wickets"
        elif "runs" in by:
            margin = f"{by['runs']} runs"
    
    # Safe list access
    dates = info.get("dates", [None])
    teams = info.get("teams", [])
    pom = info.get("player_of_match", [None])
    
    # Compress the JSON data before storing
    json_bytes = gzip.compress(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY))
    
    return MatchInfo(
        match_id=Path(filepath).stem,
        date=dates[0] if dates else None,
        venue=info.get("venue", "Unknown"),
        city=info.get("city", "Unknown"),
        teams=teams,
        series_name=event.get("name", "Unknown Series"),
        match_number=event.get("match_number", 0),
        toss_winner=toss.get("winner", "Unknown"),
        toss_decision=toss.get("decision", "Unknown"),
        winner=outcome.get("winner", "No Result"),
        margin=margin,
        method=outcome.get("method", ""),
        match_type=info.get("match_type", "Unknown"),
        season=info.get("season", "Unknown"),
        player_of_match=pom[0] if pom else None,
        original_json=json_bytes  # Now compressed
    )

def extract_deliveries(data: Dict, match_id: str) -> List[Dict]:
    """Extract ball-by-ball data with optimized list comprehension"""
    deliveries = []
    
    for inn_idx, inning in enumerate(data.get("innings", [])):
        team = inning.get("team", "Unknown")
        
        for over in inning.get("overs", []):
            over_num = over.get("over", 0)
            
            for ball_idx, ball in enumerate(over.get("deliveries", [])):
                runs = ball.get("runs", {})
                extras = ball.get("extras", {})
                wickets = ball.get("wickets", [])
                wicket_info = wickets[0] if wickets else None
                
                deliveries.append({
                    "match_id": match_id,
                    "inning": inn_idx + 1,
                    "over_num": over_num,
                    "ball_num": ball_idx + 1,
                    "batter": ball.get("batter", "Unknown"),
                    "bowler": ball.get("bowler", "Unknown"),
                    "non_striker": ball.get("non_striker", "Unknown"),
                    "runs_batter": runs.get("batter", 0),
                    "runs_extras": runs.get("extras", 0),
                    "runs_total": runs.get("total", 0),
                    "wides": extras.get("wides", 0),
                    "noballs": extras.get("noballs", 0),
                    "byes": extras.get("byes", 0),
                    "legbyes": extras.get("legbyes", 0),
                    "wicket": bool(wickets),
                    "wicket_kind": wicket_info.get("kind") if wicket_info else None,
                    "player_out": wicket_info.get("player_out") if wicket_info else None,
                    "fielders": str([f.get("name") for f in wicket_info.get("fielders", [])]) if wicket_info else None
                })
    
    return deliveries

def process_file(filepath: str) -> Tuple[Optional[MatchInfo], Optional[List[Dict]], str]:
    """Process a single JSON file (thread-safe)"""
    try:
        with open(filepath, 'rb') as f:
            data = orjson.loads(f.read())
        
        # Validate structure
        if "info" not in data or "innings" not in data:
            return None, None, f"Invalid structure: {Path(filepath).name}"

        # Extract data
        match_info = extract_match_info(data, filepath)
        deliveries = extract_deliveries(data, match_info.match_id)

        return match_info, deliveries, f"✓ {Path(filepath).name}"

    except orjson.JSONDecodeError as e:
        return None, None, f"JSON error in {Path(filepath).name}: {e}"
    except Exception as e:
        return None, None, f"Error in {Path(filepath).name}: {e}"

def batch_insert(conn: duckdb.DuckDBPyConnection, matches: List[MatchInfo], deliveries: List[Dict]) -> None:
    """Bulk insert data efficiently using executemany with manual transaction"""
    try:
        conn.execute("BEGIN TRANSACTION;")

        # Insert matches
        if matches:
            match_values = [tuple(asdict(m).values()) for m in matches]
            conn.executemany("""
                INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, match_values)

        # Insert deliveries
        if deliveries:
            delivery_values = [
                (d['match_id'], d['inning'], d['over_num'], d['ball_num'],
                 d['batter'], d['bowler'], d['non_striker'],
                 d['runs_batter'], d['runs_extras'], d['runs_total'],
                 d['wides'], d['noballs'], d['byes'], d['legbyes'],
                 d['wicket'], d['wicket_kind'], d['player_out'], d['fielders'])
                for d in deliveries
            ]
            conn.executemany("""
                INSERT INTO deliveries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, delivery_values)

        conn.execute("COMMIT;")

    except Exception as e:
        conn.execute("ROLLBACK;")
        logger.error(f"Batch insert error: {e}")
        raise



def main() -> None:
    start_time = datetime.now()
    
    # Connect to DuckDB
    conn = duckdb.connect(DB_PATH)
    conn.execute("SET threads TO 8")  # Increased threads for performance
    conn.execute("SET memory_limit = '4GB'")  # Increase memory
    init_db(conn)

    # Get all JSON files
    json_files = list(Path(DATA_DIR).glob("*.json"))
    if not json_files:
        logger.error(f"No JSON files found in {DATA_DIR}/")
        return

    logger.info(f"Found {len(json_files)} files. Starting parallel import...")

    # Process files in parallel with progress tracking
    match_batch = []
    delivery_batch = []
    processed_count = 0
    error_count = 0
    total_balls = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, fp): fp for fp in json_files}
        
        for future in as_completed(futures):
            match_info, deliveries, message = future.result()
            
            if match_info is None:
                logger.warning(message)
                error_count += 1
                continue
            
            match_batch.append(match_info)
            delivery_batch.extend(deliveries)
            processed_count += 1
            
            # Show progress every 10 files
            if processed_count % 10 == 0:
                progress_pct = (processed_count / len(json_files)) * 100
                logger.info(f"⚡ Processing: {processed_count}/{len(json_files)} ({progress_pct:.1f}%) | "
                           f"Batched: {len(delivery_batch):,} balls | Errors: {error_count}")
            
            # Batch insert when threshold reached
            if len(match_batch) >= BATCH_SIZE:
                batch_insert(conn, match_batch, delivery_batch)
                total_balls += len(delivery_batch)
                logger.info(f"💾 Saved batch: {len(match_batch)} matches, {len(delivery_batch):,} balls | "
                           f"Total: {total_balls:,} balls in DB")
                match_batch.clear()
                delivery_batch.clear()
    
    # Insert remaining batches
    if match_batch:
        batch_insert(conn, match_batch, delivery_batch)
        total_balls += len(delivery_batch)
        logger.info(f"💾 Final batch: {len(match_batch)} matches, {len(delivery_batch):,} balls")

    # Show comprehensive summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    match_count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    ball_count = conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
    series_count = conn.execute("SELECT COUNT(DISTINCT series_name) FROM matches").fetchone()[0]
    unique_players = conn.execute("SELECT COUNT(DISTINCT batter) FROM deliveries").fetchone()[0]
    date_range = conn.execute("SELECT MIN(date), MAX(date) FROM matches").fetchone()

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Import Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Database: {DB_PATH}")
    logger.info(f"Time taken: {elapsed:.2f} seconds ({len(json_files)/elapsed:.1f} files/sec)")
    logger.info(f"Files processed: {processed_count} | Errors: {error_count}")
    logger.info(f"Matches: {match_count:,}")
    logger.info(f"Deliveries: {ball_count:,}")
    logger.info(f"Series: {series_count}")
    logger.info(f"Unique players: {unique_players}")
    logger.info(f"Date range: {date_range[0]} to {date_range[1]}")
    logger.info(f"{'='*60}\n")

    conn.close()

if __name__ == "__main__":
    main()