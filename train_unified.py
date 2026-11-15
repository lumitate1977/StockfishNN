from __future__ import annotations
import argparse, csv, gzip, time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
import chess, chess.pgn
import numpy as np
import torch
import torch.nn.functional as F
from chess_ai import ACTION_SIZE, ROOT, ChessNet, encode_board, move_to_index

MATE = {"mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5"}
TACTICS = {"fork","pin","skewer","discoveredAttack","doubleCheck","hangingPiece","trappedPiece",
"deflection","attraction","interference","clearance","sacrifice","xRayAttack","exposedKing",
"zwischenzug","promotion","underPromotion","enPassant","crushing","advantage","defensiveMove",
"quietMove","capturingDefender","intermezzo"}
ENDGAMES = {"rookEndgame","bishopEndgame","knightEndgame","pawnEndgame","queenEndgame",
"queenRookEndgame","oppositeBishops","rookPawn","bishopPair","knightPair","queenVsRook"}

@dataclass
class Data:
    name: str
    x: np.ndarray
    y: np.ndarray
    value: np.ndarray
    value_weight: np.ndarray
    offsets: np.ndarray
    legal: np.ndarray
    train: np.ndarray
    val: np.ndarray

def path_of(s):
    p=Path(s).expanduser()
    return p if p.is_absolute() else ROOT/p

def split_ids(groups, fraction, rng):
    unique=np.unique(groups)
    if len(unique)<2 or fraction<=0:
        return np.arange(len(groups)), np.empty(0,dtype=np.int64)
    shuffled=rng.permutation(unique)
    nval=min(len(unique)-1,max(1,int(round(len(unique)*fraction))))
    valmask=np.isin(groups,shuffled[:nval])
    return np.flatnonzero(~valmask),np.flatnonzero(valmask)

def pack(name,x,y,v,vw,legal_lists,groups,fraction,rng):
    if not x: raise RuntimeError(f"No usable samples for {name}")
    offsets=np.zeros(len(legal_lists)+1,dtype=np.int64)
    offsets[1:]=np.cumsum([len(a) for a in legal_lists])
    legal=np.concatenate([np.asarray(a,dtype=np.int32) for a in legal_lists])
    groups=np.asarray(groups,dtype=np.int64)
    tr,va=split_ids(groups,fraction,rng)
    return Data(name,np.stack(x).astype(np.uint8,copy=False),np.asarray(y,dtype=np.int64),
        np.asarray(v,dtype=np.float32),np.asarray(vw,dtype=np.float32),offsets,legal,tr,va)

def open_pgn(p):
    return gzip.open(p,"rt",encoding="utf-8-sig",errors="replace") if p.name.lower().endswith(".pgn.gz") else p.open("r",encoding="utf-8-sig",errors="replace")

def load_pgn(source,max_positions,min_elo,fraction,rng):
    files=([source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file() and p.name.lower().endswith((".pgn",".pgn.gz")))) if source.exists() else []
    files=[p for p in files if p.name.lower().endswith((".pgn",".pgn.gz"))]
    if not files: raise FileNotFoundError(f"No PGN files found: {source}")
    x=[];y=[];v=[];vw=[];ll=[];groups=[];seen=used=0
    for p in files:
        print(f"  PGN: {p.name}")
        with open_pgn(p) as f:
            while len(x)<max_positions:
                g=chess.pgn.read_game(f)
                if g is None: break
                seen+=1; result=g.headers.get("Result","*")
                if result not in {"1-0","0-1","1/2-1/2"}: continue
                if min_elo:
                    try:
                        if min(int(g.headers.get("WhiteElo","0")),int(g.headers.get("BlackElo","0")))<min_elo: continue
                    except ValueError: continue
                white=1.0 if result=="1-0" else -1.0 if result=="0-1" else 0.0
                b=g.board(); added=0
                for m in g.mainline_moves():
                    lm=list(b.legal_moves)
                    if m not in lm: break
                    x.append(encode_board(b)); y.append(move_to_index(m))
                    v.append(white if b.turn==chess.WHITE else -white); vw.append(1.0)
                    ll.append([move_to_index(a) for a in lm]); groups.append(seen)
                    b.push(m); added+=1
                    if len(x)>=max_positions: break
                used+=int(added>0)
        if len(x)>=max_positions: break
    print(f"  Human PGN: {len(x):,} positions from {used:,} games ({seen:,} read)")
    return pack("human_pgn",x,y,v,vw,ll,groups,fraction,rng)

def category(themes):
    if themes&MATE: return "mate"
    if themes&ENDGAMES: return "endgame"
    if themes&TACTICS: return "tactics"
    return None

def parse_puzzle(row,skip):
    fen=(row.get("FEN") or row.get("fen") or "").strip()
    text=(row.get("Moves") or row.get("moves") or "").strip()
    cat=category(set((row.get("Themes") or row.get("themes") or "").split()))
    if not fen or not text or not cat: return None
    try:
        b=chess.Board(fen); moves=[chess.Move.from_uci(t) for t in text.split()]
        if skip:
            if not moves or moves[0] not in b.legal_moves: return None
            b.push(moves.pop(0))
        solver=b.turn; samples=[]
        for m in moves:
            lm=list(b.legal_moves)
            if m not in lm: return None
            if b.turn==solver: samples.append((encode_board(b),move_to_index(m),[move_to_index(a) for a in lm]))
            b.push(m)
        if not samples or (cat=="mate" and not b.is_checkmate()): return None
        return cat,samples
    except (ValueError,AssertionError): return None

def load_puzzles(path,max_rows,skip,fraction,rng):
    buckets={k:([],[],[],[],[],[]) for k in ("mate_puzzles","tactics_puzzles","endgame_puzzles")}
    seen=used=0
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or not {"FEN","Moves","Themes"}.issubset(reader.fieldnames):
            raise RuntimeError("Puzzle CSV must have FEN, Moves, Themes columns.")
        for row in reader:
            seen+=1; parsed=parse_puzzle(row,skip)
            if parsed:
                cat,samples=parsed
                key={"mate":"mate_puzzles","tactics":"tactics_puzzles","endgame":"endgame_puzzles"}[cat]
                bucket=buckets[key]
                for pos,act,legal in samples:
                    bucket[0].append(pos);bucket[1].append(act);bucket[2].append(0.0)
                    bucket[3].append(0.0);bucket[4].append(legal);bucket[5].append(seen)
                used+=1
            if max_rows and seen>=max_rows: break
    print(f"  Puzzle CSV: scanned {seen:,} rows; accepted {used:,} puzzles")
    out={}
    for name,(x,y,v,vw,ll,groups) in buckets.items():
        if x:
            out[name]=pack(name,x,y,v,vw,ll,groups,fraction,rng)
            print(f"  {name}: {len(x):,} positions")
    if not out: raise RuntimeError("No supported mate/tactics/endgame puzzles found in scanned CSV rows.")
    return out

def load_replay(path,fraction,rng):
    if not path.is_file(): print(f"  Replay missing, skipped: {path}"); return None
    with np.load(path,allow_pickle=False) as d:
        required={"positions","teacher","values","chosen","legal_offsets","legal_actions"}
        if not required.issubset(d.files): raise RuntimeError(f"Replay missing arrays: {sorted(required-set(d.files))}")
        x=d["positions"]; y=d["teacher"]; v=d["values"]; off=d["legal_offsets"]; legal=d["legal_actions"]
        n=len(x)
        if x.shape!=(n,18,8,8) or len(y)!=n or len(v)!=n or len(off)!=n+1:
            raise RuntimeError("Replay array shapes/lengths do not match.")
        if off[0]!=0 or off[-1]!=len(legal) or np.any(np.diff(off)<=0):
            raise RuntimeError("Replay legal_offsets are not valid.")
        if np.any((y<0)|(y>=ACTION_SIZE)) or np.any((legal<0)|(legal>=ACTION_SIZE)) or not np.all(np.isfinite(v)):
            raise RuntimeError("Replay contains invalid action indices or non-finite values.")
        lists=[]
        for i in range(n):
            row=legal[off[i]:off[i+1]]
            if y[i] not in row: raise RuntimeError(f"Replay teacher action is not legal at row {i}.")
            lists.append(row.astype(np.int32,copy=False).tolist())
    print(f"  Replay: {n:,} positions; offsets and teacher legality verified")
    return pack("replay",[p for p in x],y.tolist(),v.tolist(),np.ones(n,dtype=np.float32).tolist(),
                lists,np.arange(n),fraction,rng)

def mask_for(ds,ids,device):
    starts,ends=ds.offsets[ids],ds.offsets[ids+1]
    rows=np.repeat(np.arange(len(ids),dtype=np.int64),ends-starts)
    actions=np.concatenate([ds.legal[a:b] for a,b in zip(starts,ends)])
    mask=np.zeros((len(ids),ACTION_SIZE),dtype=np.bool_);mask[rows,actions]=True
    return torch.from_numpy(mask).to(device)

def train_batch(model,opt,ds,ids,device,value_weight):
    x=torch.from_numpy(ds.x[ids].astype(np.float32,copy=False)).to(device)
    y=torch.from_numpy(ds.y[ids]).to(device=device,dtype=torch.long)
    v=torch.from_numpy(ds.value[ids]).to(device); vw=torch.from_numpy(ds.value_weight[ids]).to(device)
    logits,pred=model(x); policy=F.cross_entropy(logits.masked_fill(~mask_for(ds,ids,device),-1e9),y)
    denom=vw.sum()
    value=((pred-v).square()*vw).sum()/denom if denom.item()>0 else pred.sum()*0
    loss=policy+value_weight*value
    opt.zero_grad(set_to_none=True);loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
    return float(policy.detach()),float(value.detach())

@torch.no_grad()
def evaluate(model,ds,ids,device,batch):
    if not len(ids): return None
    model.eval();correct=total=0;err=weight_sum=0.0
    for start in range(0,len(ids),batch):
        ix=ids[start:start+batch]
        x=torch.from_numpy(ds.x[ix].astype(np.float32,copy=False)).to(device)
        logits,pred=model(x);guess=logits.masked_fill(~mask_for(ds,ix,device),-1e9).argmax(1).cpu().numpy()
        correct+=int((guess==ds.y[ix]).sum());total+=len(ix)
        w=ds.value_weight[ix]
        if np.any(w>0):
            tv=torch.from_numpy(ds.value[ix]).to(device);tw=torch.from_numpy(w).to(device)
            err+=float(((pred-tv).abs()*tw).sum());weight_sum+=float(tw.sum())
    return correct/total,(err/weight_sum if weight_sum else None)

def main():
    p=argparse.ArgumentParser(description="Balanced mixed-source training for one StockfishNN model.")
    p.add_argument("--base-checkpoint",default="models/human_pretrained.pt")
    p.add_argument("--output",default="models/unified.pt")
    p.add_argument("--pgn-dir",default="data/pgn");p.add_argument("--puzzles",default="data/puzzles/lichess_db_puzzle.csv")
    p.add_argument("--replay",default="data/replay_buffer.npz");p.add_argument("--no-replay",action="store_true")
    p.add_argument("--puzzle-rows",type=int,default=100000);p.add_argument("--max-pgn-positions",type=int,default=100000)
    p.add_argument("--min-elo",type=int,default=0);p.add_argument("--samples-per-source",type=int,default=10000)
    p.add_argument("--epochs",type=int,default=3);p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--learning-rate",type=float,default=0.00001);p.add_argument("--validation-fraction",type=float,default=0.1)
    p.add_argument("--value-weight",type=float,default=0.25);p.add_argument("--torch-threads",type=int,default=4)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--no-skip-setup-move",action="store_true")
    p.add_argument("--overwrite",action="store_true");a=p.parse_args()
    if min(a.epochs,a.batch_size,a.samples_per_source,a.max_pgn_positions)<=0 or a.puzzle_rows<0 or a.min_elo<0:
        p.error("epochs, batch-size, samples-per-source and max-pgn-positions must be positive; puzzle-rows/min-elo nonnegative")
    if a.learning_rate<=0 or a.value_weight<0 or not 0<=a.validation_fraction<0.5: p.error("Invalid learning-rate, value-weight, or validation-fraction")
    base=path_of(a.base_checkpoint);out=path_of(a.output)
    protected={path_of(s).resolve() for s in ("models/latest.pt","models/human_pretrained.pt","models/mate_pretrained.pt")}
    if out.resolve() in protected or out.resolve()==base.resolve(): p.error("Output must not overwrite a source checkpoint.")
    if out.exists() and not a.overwrite: p.error(f"Output exists: {out}; choose another name or pass --overwrite.")
    if not base.is_file(): raise FileNotFoundError(f"Base checkpoint not found: {base}")
    rng=np.random.default_rng(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(max(1,a.torch_threads));device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; starting from {base}")
    datasets=[load_pgn(path_of(a.pgn_dir),a.max_pgn_positions,a.min_elo,a.validation_fraction,rng)]
    datasets.extend(load_puzzles(path_of(a.puzzles),a.puzzle_rows,not a.no_skip_setup_move,a.validation_fraction,rng).values())
    if not a.no_replay:
        replay=load_replay(path_of(a.replay),a.validation_fraction,rng)
        if replay is not None: datasets.append(replay)
    if len(datasets)<2: raise RuntimeError("At least two data sources are required.")
    ckpt=torch.load(base,map_location=device,weights_only=False)
    if "model_state" not in ckpt: raise RuntimeError("Base file is not a StockfishNN checkpoint.")
    model=ChessNet().to(device);model.load_state_dict(ckpt["model_state"])
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4)
    print("\nBalanced datasets:")
    for ds in datasets: print(f"  {ds.name}: {len(ds.train):,} train / {len(ds.val):,} validation")
    print(f"Sampling {a.samples_per_source:,} positions per source each epoch.")
    start=time.time()
    for epoch in range(1,a.epochs+1):
        rng.shuffle(datasets);losses={}
        for ds in datasets:
            if not len(ds.train): continue
            ids=rng.choice(ds.train,size=a.samples_per_source,replace=a.samples_per_source>len(ds.train));rng.shuffle(ids)
            pl=[];vl=[]
            model.train()
            for i in range(0,len(ids),a.batch_size):
                q,r=train_batch(model,opt,ds,ids[i:i+a.batch_size],device,a.value_weight);pl.append(q);vl.append(r)
            losses[ds.name]=(np.mean(pl),np.mean(vl))
        print(f"\nEpoch {epoch}/{a.epochs}")
        for ds in datasets:
            if ds.name in losses: print(f"  {ds.name}: policy {losses[ds.name][0]:.5f}, value {losses[ds.name][1]:.5f}")
        for ds in datasets:
            metrics=evaluate(model,ds,ds.val,device,a.batch_size)
            if metrics:
                acc,mae=metrics;extra=f", value MAE {mae:.4f}" if mae is not None else ""
                print(f"  validation {ds.name}: legal top-1 {acc:.2%}{extra}")
    out.parent.mkdir(parents=True,exist_ok=True)
    meta=dict(ckpt.get("metadata",{}));meta.update({"pretraining":"unified_mixed_sources","base_checkpoint":str(base),
        "epochs":a.epochs,"samples_per_source_per_epoch":a.samples_per_source,"learning_rate":a.learning_rate,
        "sources":[d.name for d in datasets],"source_train_positions":{d.name:int(len(d.train)) for d in datasets},
        "source_validation_positions":{d.name:int(len(d.val)) for d in datasets},"architecture":"ChessNet-18plane-policy-value-v1"})
    torch.save({"model_state":model.state_dict(),"optimizer_state":opt.state_dict(),"metadata":meta},out)
    print(f"\nUnified checkpoint saved: {out}; elapsed {time.time()-start:.1f}s")
    print("All input checkpoints and datasets were read-only.")

if __name__=="__main__": main()
