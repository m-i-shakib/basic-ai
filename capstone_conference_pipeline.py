# COMPLETE CAPSTONE CONFERENCE PAPER COLAB PIPELINE
# Project: Agentic Explainable Agricultural Decision Support System using Multi-Agent AI, RAG and Machine Learning
# Public-dataset preliminary experiment for current semester

# ===================== CONFIG =====================
SEED=42
DATASET_REPO='CGIAR/RAG-Chunk-Analysis'
MODEL_NAME='Qwen/Qwen3-4B-Instruct-2507'
BI_ENCODER_NAME='sentence-transformers/all-MiniLM-L6-v2'
CROSS_ENCODER_NAME='cross-encoder/ms-marco-MiniLM-L-6-v2'
TOP_K=5
MIN_COVERAGE=0.90
MAX_GENERATION_QUERIES=None
MAX_CONTEXT_CHARS=1600
MAX_INPUT_TOKENS=4096
MAX_NEW_TOKENS=110
DRIVE_DIR='/content/drive/MyDrive/Capstone_Conference_Paper'

# ===================== INSTALL + SETUP =====================
import subprocess,sys
subprocess.run([sys.executable,'-m','pip','install','-q','-U','transformers>=4.51.0','accelerate','bitsandbytes','sentence-transformers','huggingface_hub','pandas','numpy','openpyxl','xlrd','scikit-learn','matplotlib','tqdm'],check=True)

import os,re,json,time,random,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer,CrossEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import ndcg_score
from google.colab import drive

warnings.filterwarnings('ignore')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print('CUDA available:',torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError('GPU required. Select Runtime > Change runtime type > T4 GPU, then Run all again.')
print('GPU:',torch.cuda.get_device_name(0))
drive.mount('/content/drive',force_remount=False)
SAVE_DIR=Path(DRIVE_DIR); SAVE_DIR.mkdir(parents=True,exist_ok=True)
print('Permanent output folder:',SAVE_DIR)

# ===================== DATA DOWNLOAD + PREPROCESS =====================
DATA_DIR=Path('/content/CGIAR_RAG_Chunk_Analysis')
snapshot_download(repo_id=DATASET_REPO,repo_type='dataset',local_dir=str(DATA_DIR))
files=sorted(list(DATA_DIR.rglob('*.xlsx'))+list(DATA_DIR.rglob('*.xls')))
frames=[]
for f in files:
    try:
        for sheet,df in pd.read_excel(f,sheet_name=None).items():
            if df is None or df.empty: continue
            t=df.copy(); t['source_file']=f.name; t['source_sheet']=str(sheet); frames.append(t)
    except Exception as e:
        print('Skipped:',f.name,'|',e)
if not frames: raise RuntimeError('No readable Excel sheets found.')
raw_df=pd.concat(frames,ignore_index=True,sort=False)

def clean_text(x):
    if pd.isna(x): return None
    s=str(x).strip(); return None if (not s or s.lower()=='nan') else s

def relevance_label(x):
    if pd.isna(x): return np.nan
    s=str(x).strip().lower()
    if 'irrelevant' in s or 'not relevant' in s: return 0
    if 'relevant' in s: return 1
    if s in {'yes','y','1','true'}: return 1
    if s in {'no','n','0','false'}: return 0
    try:
        v=float(x)
        if v in (0,1): return int(v)
    except Exception: pass
    return np.nan

chunk_cols=[]
for col in raw_df.columns:
    m=re.fullmatch(r'Chunk\s+(\d+)',str(col).strip())
    if m: chunk_cols.append((int(m.group(1)),col))
chunk_cols=sorted(chunk_cols)
if not chunk_cols: raise ValueError('No Chunk N columns found.')

rows=[]
for row_id,row in raw_df.iterrows():
    query=clean_text(row.get('User Query'))
    if query is None: continue
    response=clean_text(row.get('Response'))
    for chunk_number,chunk_col in chunk_cols:
        suffix='' if chunk_number==1 else f'.{chunk_number-1}'
        chunk=clean_text(row.get(chunk_col))
        if chunk is None: continue
        rel_col=f'Relevant/ Irrelevant{suffix}'
        rows.append({'original_row_id':row_id,'query':query,'original_response':response,'chunk_number':chunk_number,'chunk':chunk,'source':clean_text(row.get(f'Source{suffix}')),'page':row.get(f'Page{suffix}') if f'Page{suffix}' in raw_df.columns else np.nan,'relevance_raw':row.get(rel_col) if rel_col in raw_df.columns else np.nan,'relevance_label':relevance_label(row.get(rel_col) if rel_col in raw_df.columns else np.nan),'relevant_portion':clean_text(row.get(f'Relevant_portion{suffix}')) if f'Relevant_portion{suffix}' in raw_df.columns else None,'source_file':row.get('source_file'),'source_sheet':row.get('source_sheet')})

rag_df=pd.DataFrame(rows).drop_duplicates(subset=['query','chunk','chunk_number']).reset_index(drop=True)
rag_df.to_csv(SAVE_DIR/'CGIAR_RAG_Long_Format.csv',index=False)
eval_df=rag_df.dropna(subset=['query','chunk','relevance_label']).copy()
eval_df=eval_df[eval_df['relevance_label'].isin([0,1])].drop_duplicates(subset=['query','chunk'])
valid=eval_df.groupby('query')['relevance_label'].sum()
eval_df=eval_df[eval_df['query'].isin(valid[valid>0].index)].copy()
print('Raw:',raw_df.shape,'Long:',rag_df.shape,'Judged pairs:',len(eval_df),'Queries:',eval_df['query'].nunique())

# ===================== R0 / R1 / R2 RETRIEVAL =====================
bi_encoder=SentenceTransformer(BI_ENCODER_NAME)
unique_queries=eval_df['query'].unique().tolist(); unique_chunks=eval_df['chunk'].unique().tolist()
query_embeddings=bi_encoder.encode(unique_queries,batch_size=64,show_progress_bar=True,normalize_embeddings=True)
chunk_embeddings=bi_encoder.encode(unique_chunks,batch_size=64,show_progress_bar=True,normalize_embeddings=True)
qmap=dict(zip(unique_queries,query_embeddings)); cmap=dict(zip(unique_chunks,chunk_embeddings))
eval_df['embedding_similarity']=[float(np.dot(qmap[q],cmap[c])) for q,c in zip(eval_df['query'],eval_df['chunk'])]

cross_encoder=CrossEncoder(CROSS_ENCODER_NAME)
eval_df['cross_encoder_score']=cross_encoder.predict(list(zip(eval_df['query'],eval_df['chunk'])),batch_size=32,show_progress_bar=True)

def ranking_metrics(group,ranking_col,ascending):
    ranked=group.sort_values(ranking_col,ascending=ascending)
    labels=ranked['relevance_label'].astype(int).tolist(); total_relevant=sum(labels); k=min(TOP_K,len(labels)); top=labels[:k]
    rr=0.0
    for i,label in enumerate(labels,start=1):
        if label==1: rr=1/i; break
    ndcg=ndcg_score(np.array([labels]),np.array([list(range(len(labels),0,-1))]),k=k) if len(labels)>1 else float(labels[0])
    return {'Hit@1':labels[0],'MRR':rr,f'Precision@{TOP_K}':sum(top)/k,f'Recall@{TOP_K}':sum(top)/total_relevant,f'NDCG@{TOP_K}':ndcg}

def evaluate_ranking(df,ranking_col,ascending):
    out=[]
    for query,group in df.groupby('query'):
        row=ranking_metrics(group,ranking_col,ascending); row['query']=query; out.append(row)
    return pd.DataFrame(out)

r0_df=evaluate_ranking(eval_df,'chunk_number',True)
r1_df=evaluate_ranking(eval_df,'embedding_similarity',False)
r2_df=evaluate_ranking(eval_df,'cross_encoder_score',False)
metric_cols=['Hit@1','MRR',f'Precision@{TOP_K}',f'Recall@{TOP_K}',f'NDCG@{TOP_K}']
retrieval_summary=pd.DataFrame({'R0 Original Ranking':r0_df[metric_cols].mean(),'R1 MiniLM Bi-Encoder':r1_df[metric_cols].mean(),'R2 Cross-Encoder':r2_df[metric_cols].mean()})
print('\nR0 / R1 / R2 RETRIEVAL SUMMARY')
print(retrieval_summary.round(4))
eval_df.to_csv(SAVE_DIR/'R2_all_scored_query_chunk_pairs.csv',index=False)
r0_df.to_csv(SAVE_DIR/'R0_original_ranking_metrics.csv',index=False); r1_df.to_csv(SAVE_DIR/'R1_minilm_reranking_metrics.csv',index=False); r2_df.to_csv(SAVE_DIR/'R2_cross_encoder_metrics.csv',index=False)
retrieval_summary.to_csv(SAVE_DIR/'R0_R1_R2_comparison.csv')

# ===================== CONFIDENCE ESTIMATOR =====================
gate_df=eval_df.copy(); gate_df['original_rank_signal']=1.0/gate_df['chunk_number']
gate_df['bi_rank']=gate_df.groupby('query')['embedding_similarity'].rank(method='first',ascending=False)
gate_df['cross_rank']=gate_df.groupby('query')['cross_encoder_score'].rank(method='first',ascending=False)
gate_df['bi_rank_signal']=1.0/gate_df['bi_rank']; gate_df['cross_rank_signal']=1.0/gate_df['cross_rank']
FEATURES=['original_rank_signal','embedding_similarity','cross_encoder_score','bi_rank_signal','cross_rank_signal']; TARGET='relevance_label'
gate_df=gate_df.dropna(subset=FEATURES+[TARGET]).copy()
split1=GroupShuffleSplit(n_splits=1,test_size=.30,random_state=SEED); train_idx,temp_idx=next(split1.split(gate_df,groups=gate_df['query']))
train_df=gate_df.iloc[train_idx].copy(); temp_df=gate_df.iloc[temp_idx].copy()
split2=GroupShuffleSplit(n_splits=1,test_size=.50,random_state=SEED); val_idx,test_idx=next(split2.split(temp_df,groups=temp_df['query']))
val_df=temp_df.iloc[val_idx].copy(); test_df=temp_df.iloc[test_idx].copy()
train_q=set(train_df['query']); val_q=set(val_df['query']); test_q=set(test_df['query'])
assert not(train_q&val_q or train_q&test_q or val_q&test_q),'Query leakage detected.'
gate_model=Pipeline([('scaler',StandardScaler()),('classifier',LogisticRegression(class_weight='balanced',max_iter=2000,random_state=SEED))])
gate_model.fit(train_df[FEATURES],train_df[TARGET].astype(int))
val_prob=gate_model.predict_proba(val_df[FEATURES])[:,1]; test_prob=gate_model.predict_proba(test_df[FEATURES])[:,1]
print('Train/Val/Test queries:',train_df['query'].nunique(),val_df['query'].nunique(),test_df['query'].nunique())

# ===================== VALIDATION THRESHOLD + TEST GATE =====================
val_work=val_df.copy(); val_work['evidence_confidence']=val_prob
curve=[]
for threshold in np.arange(.20,.91,.05):
    query_rows=[]
    for query,group in val_work.groupby('query'):
        accepted=group[group['evidence_confidence']>=threshold]; covered=int(len(accepted)>0); total_relevant=group['relevance_label'].sum()
        query_rows.append({'covered':covered,'precision':accepted['relevance_label'].mean() if covered else np.nan,'recall':accepted['relevance_label'].sum()/total_relevant if total_relevant>0 else np.nan,'top1':int(accepted.sort_values('evidence_confidence',ascending=False).iloc[0]['relevance_label']) if covered else np.nan})
    temp=pd.DataFrame(query_rows); coverage=temp['covered'].mean()
    curve.append({'threshold':round(float(threshold),2),'coverage':coverage,'abstention_rate':1-coverage,'selective_precision':temp['precision'].mean(skipna=True),'selective_recall':temp['recall'].mean(skipna=True),'top1_relevance':temp['top1'].mean(skipna=True)})
coverage_curve=pd.DataFrame(curve); eligible=coverage_curve[coverage_curve['coverage']>=MIN_COVERAGE].copy()
if eligible.empty: raise RuntimeError('No threshold satisfies minimum coverage.')
best_row=eligible.sort_values(['selective_precision','top1_relevance'],ascending=False).iloc[0]; FINAL_GATE_THRESHOLD=float(best_row['threshold'])
print('FINAL GATE THRESHOLD:',FINAL_GATE_THRESHOLD)

test_df=test_df.copy(); test_df['evidence_confidence']=test_prob; test_df['gate_decision']=np.where(test_df['evidence_confidence']>=FINAL_GATE_THRESHOLD,'ACCEPT','REJECT')
query_rows=[]
for query,group in test_df.groupby('query'):
    total_relevant=int(group['relevance_label'].sum()); baseline=group.sort_values('chunk_number').head(TOP_K); accepted=group[group['evidence_confidence']>=FINAL_GATE_THRESHOLD].sort_values('evidence_confidence',ascending=False).head(TOP_K); covered=int(len(accepted)>0)
    query_rows.append({'query':query,'baseline_precision':baseline['relevance_label'].mean(),'baseline_recall':baseline['relevance_label'].sum()/total_relevant,'baseline_top1':int(baseline.iloc[0]['relevance_label']),'gate_precision':accepted['relevance_label'].mean() if covered else np.nan,'gate_recall':accepted['relevance_label'].sum()/total_relevant if covered else 0.0,'gate_top1':int(accepted.iloc[0]['relevance_label']) if covered else np.nan,'covered':covered,'abstained':1-covered,'accepted_chunks':len(accepted)})
gate_query_df=pd.DataFrame(query_rows)
gate_summary=pd.DataFrame({'Metric':['Evidence Precision','Evidence Recall','Top-1 Relevance','Query Coverage','Abstention Rate'],'Original Retrieval':[gate_query_df['baseline_precision'].mean(),gate_query_df['baseline_recall'].mean(),gate_query_df['baseline_top1'].mean(),1.0,0.0],'Proposed Confidence Gate':[gate_query_df['gate_precision'].mean(skipna=True),gate_query_df['gate_recall'].mean(),gate_query_df['gate_top1'].mean(skipna=True),gate_query_df['covered'].mean(),gate_query_df['abstained'].mean()]})
print('\nFINAL EVIDENCE GATE SUMMARY')
print(gate_summary.round(4))
coverage_curve.to_csv(SAVE_DIR/'P1_validation_confidence_coverage_curve.csv',index=False); test_df.to_csv(SAVE_DIR/'P1_chunk_level_confidence_results.csv',index=False); gate_query_df.to_csv(SAVE_DIR/'P1_query_level_gate_results.csv',index=False); gate_summary.to_csv(SAVE_DIR/'P1_gate_vs_original_summary.csv',index=False)

# Figures
ax=retrieval_summary.plot(kind='bar',figsize=(10,5)); ax.set_ylim(0,1); ax.set_title('R0 vs R1 vs R2 Retrieval Comparison'); plt.tight_layout(); plt.savefig(SAVE_DIR/'Figure_R0_R1_R2_Retrieval_Comparison.png',dpi=220); plt.show()
fig,ax=plt.subplots(figsize=(8,5))
for col,label in [('selective_precision','Precision'),('selective_recall','Recall'),('coverage','Coverage')]: ax.plot(coverage_curve['threshold'],coverage_curve[col],marker='o',label=label)
ax.axvline(FINAL_GATE_THRESHOLD,linestyle='--',label=f'Selected={FINAL_GATE_THRESHOLD:.2f}'); ax.set_ylim(0,1.02); ax.set_title('Validation Confidence-Coverage Trade-off'); ax.legend(); plt.tight_layout(); plt.savefig(SAVE_DIR/'Figure_Confidence_Coverage_Tradeoff.png',dpi=220); plt.show()

# ===================== QWEN 4-BIT =====================
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
bnb_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
llm=AutoModelForCausalLM.from_pretrained(MODEL_NAME,quantization_config=bnb_config,device_map='auto'); llm.eval()
print('Qwen device:',next(llm.parameters()).device)
if 'cuda' not in str(next(llm.parameters()).device): raise RuntimeError('Qwen did not load on GPU.')
SYSTEM_PROMPT='''You are an agricultural advisory assistant. Answer the farmer question using ONLY the supplied evidence. Do not invent facts or use unsupported outside knowledge. If evidence is insufficient, say: "Insufficient evidence to provide a reliable answer." Keep the answer concise and practical. Cite supporting evidence using [E1], [E2], etc.'''

def generate_answer(query,contexts):
    contexts=[str(x).strip()[:MAX_CONTEXT_CHARS] for x in contexts[:TOP_K]]; evidence_text='\n\n'.join(f'[E{i+1}] {text}' for i,text in enumerate(contexts))
    messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':f'Farmer Question:\n{query}\n\nAgricultural Evidence:\n{evidence_text}\n\nProvide an evidence-grounded answer.'}]
    inputs=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True); inputs={k:v.to(llm.device) for k,v in inputs.items()}
    if inputs['input_ids'].shape[1]>MAX_INPUT_TOKENS:
        inputs['input_ids']=inputs['input_ids'][:,-MAX_INPUT_TOKENS:]
        if 'attention_mask' in inputs: inputs['attention_mask']=inputs['attention_mask'][:,-MAX_INPUT_TOKENS:]
    input_len=inputs['input_ids'].shape[1]; start=time.perf_counter()
    with torch.inference_mode(): output=llm.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0,input_len:],skip_special_tokens=True).strip(),time.perf_counter()-start,input_len

# ===================== FULL G1 vs G2 WITH CHECKPOINT =====================
GEN_PATH=SAVE_DIR/'G1_G2_Qwen_RAG_Full_Test.csv'; queries=test_df['query'].drop_duplicates().tolist()
if MAX_GENERATION_QUERIES is not None: queries=queries[:MAX_GENERATION_QUERIES]
if GEN_PATH.exists():
    previous=pd.read_csv(GEN_PATH); completed=set(previous['query'].dropna().astype(str)); generation_rows=previous.to_dict('records'); print('Resuming',len(completed),'completed queries.')
else: completed=set(); generation_rows=[]
remaining=[q for q in queries if q not in completed]
print('Target:',len(queries),'Remaining:',len(remaining))
for query in tqdm(remaining,desc='Qwen Original vs Gated RAG'):
    group=test_df[test_df['query']==query].copy(); original_group=group.sort_values('chunk_number').head(TOP_K); original_context=original_group['chunk'].tolist(); original_answer,original_latency,original_tokens=generate_answer(query,original_context)
    gated_group=group[group['evidence_confidence']>=FINAL_GATE_THRESHOLD].sort_values('evidence_confidence',ascending=False).head(TOP_K); gated_context=gated_group['chunk'].tolist()
    if len(gated_context)==0: gated_answer='ABSTAIN: No evidence passed the confidence threshold.'; gated_latency=0.0; gated_tokens=0; gate_abstained=1
    else: gated_answer,gated_latency,gated_tokens=generate_answer(query,gated_context); gate_abstained=0
    dataset_response=group['original_response'].dropna().astype(str).iloc[0] if ('original_response' in group.columns and group['original_response'].notna().any()) else None
    generation_rows.append({'query':query,'original_context_count':len(original_context),'gated_context_count':len(gated_context),'original_rag_answer':original_answer,'gated_rag_answer':gated_answer,'original_latency_s':original_latency,'gated_latency_s':gated_latency,'original_input_tokens':original_tokens,'gated_input_tokens':gated_tokens,'gate_abstained':gate_abstained,'dataset_response':dataset_response,'original_context':'\n\n'.join(original_context),'gated_context':'\n\n'.join(gated_context)})
    pd.DataFrame(generation_rows).to_csv(GEN_PATH,index=False)
generation_df=pd.DataFrame(generation_rows)

# ===================== END-TO-END PROXY EVALUATION =====================
def split_sentences(text): return [x.strip() for x in re.split(r'(?<=[.!?])\s+',str(text).strip()) if x.strip()]
def citation_present(text): return int(bool(re.search(r'\[E\d+\]',str(text))))
def support_proxy(answer,context):
    s=split_sentences(answer); chunks=[x.strip() for x in str(context).split('\n\n') if x.strip()]
    if not s or not chunks: return np.nan
    a=bi_encoder.encode(s,normalize_embeddings=True); c=bi_encoder.encode(chunks,normalize_embeddings=True); return float((a@c.T).max(axis=1).mean())
def text_similarity(a,b):
    if pd.isna(a) or pd.isna(b): return np.nan
    a=str(a).strip(); b=str(b).strip()
    if not a or not b: return np.nan
    emb=bi_encoder.encode([a,b],normalize_embeddings=True); return float(np.dot(emb[0],emb[1]))

generation_eval=generation_df.copy(); generation_eval['original_citation_present']=generation_eval['original_rag_answer'].apply(citation_present); generation_eval['gated_citation_present']=generation_eval['gated_rag_answer'].apply(citation_present)
original_support=[]; gated_support=[]; original_similarity=[]; gated_similarity=[]
for _,row in tqdm(generation_eval.iterrows(),total=len(generation_eval),desc='Evaluating generations'):
    original_support.append(support_proxy(row['original_rag_answer'],row['original_context'])); gated_support.append(np.nan if int(row['gate_abstained']) else support_proxy(row['gated_rag_answer'],row['gated_context'])); original_similarity.append(text_similarity(row['original_rag_answer'],row['dataset_response'])); gated_similarity.append(np.nan if int(row['gate_abstained']) else text_similarity(row['gated_rag_answer'],row['dataset_response']))
generation_eval['original_support_proxy']=original_support; generation_eval['gated_support_proxy']=gated_support; generation_eval['original_dataset_response_similarity']=original_similarity; generation_eval['gated_dataset_response_similarity']=gated_similarity
end_to_end_summary=pd.DataFrame({'Metric':['Mean Context Count','Mean Latency (s)','Citation Presence Rate','Semantic Evidence-Support Proxy','Dataset-Response Similarity Proxy','Abstention Rate'],'G1 Original RAG':[generation_eval['original_context_count'].mean(),generation_eval['original_latency_s'].mean(),generation_eval['original_citation_present'].mean(),generation_eval['original_support_proxy'].mean(skipna=True),generation_eval['original_dataset_response_similarity'].mean(skipna=True),0.0],'G2 Confidence-Gated RAG':[generation_eval['gated_context_count'].mean(),generation_eval['gated_latency_s'].mean(),generation_eval['gated_citation_present'].mean(),generation_eval['gated_support_proxy'].mean(skipna=True),generation_eval['gated_dataset_response_similarity'].mean(skipna=True),generation_eval['gate_abstained'].mean()]})
print('\nEND-TO-END SUMMARY'); print(end_to_end_summary.round(4))
generation_eval.to_csv(SAVE_DIR/'G1_G2_Qwen_RAG_Evaluated.csv',index=False); end_to_end_summary.to_csv(SAVE_DIR/'G1_G2_End_to_End_Summary.csv',index=False)
plot_part=end_to_end_summary[end_to_end_summary['Metric'].isin(['Citation Presence Rate','Semantic Evidence-Support Proxy','Dataset-Response Similarity Proxy'])].set_index('Metric'); ax=plot_part.plot(kind='bar',figsize=(9,5)); ax.set_ylim(0,1); ax.set_title('End-to-End Original RAG vs Confidence-Gated RAG'); plt.xticks(rotation=15,ha='right'); plt.tight_layout(); plt.savefig(SAVE_DIR/'Figure_End_to_End_RAG_Comparison.png',dpi=220); plt.show()
summary_lines=['CAPSTONE CONFERENCE PAPER — AUTOMATIC RESULT SUMMARY','='*72,f'Dataset: {DATASET_REPO}',f'Judged query-chunk pairs: {len(eval_df)}',f'Evaluation queries: {eval_df["query"].nunique()}',f'Train/Validation/Test queries: {train_df["query"].nunique()} / {val_df["query"].nunique()} / {test_df["query"].nunique()}','','R0/R1/R2:',retrieval_summary.round(4).to_string(),'',f'Final confidence threshold: {FINAL_GATE_THRESHOLD:.2f}','','Evidence Gate Test:',gate_summary.round(4).to_string(index=False),'','End-to-End Qwen:',end_to_end_summary.round(4).to_string(index=False),'','NOTE: Dataset-response similarity is a proxy, not gold-answer accuracy.','NOTE: Semantic evidence-support is an embedding-based proxy, not a true faithfulness metric.']
summary_text='\n'.join(summary_lines); (SAVE_DIR/'Conference_Paper_Automatic_Result_Summary.txt').write_text(summary_text,encoding='utf-8')
print(summary_text); print('\nALL DONE. Results saved in:',SAVE_DIR)
