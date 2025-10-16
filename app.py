# app.py (최종 완성본)

from flask import Flask, request, jsonify, render_template, Response, send_from_directory
import os
import traceback
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid
import zipfile # ZIP 파일 생성을 위해 추가
from io import BytesIO # 메모리에서 파일 처리를 위해 추가

# analyze_track.py에서 필요한 함수를 가져옵니다.
from analyze_track import analyze_one, asdict

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 진행 상황을 저장할 '진행 상황판'
analysis_progress = {}

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# === 추가된 부분: 프로젝트 파일 전체 다운로드 기능 ===
@app.route('/download_project_files')
def download_project_files():
    # 메모리 상에서 ZIP 파일을 생성합니다.
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        # 1. app.py (현재 실행 중인 이 파일)
        zf.write('app.py')
        # 2. analyze_track.py
        zf.write('analyze_track.py')
        # 3. templates/index.html (사용자가 보고 있는 이 웹페이지)
        zf.write('templates/index.html', arcname='templates/index.html')
        # 4. (선택) requirements.txt (필요한 라이브러리 목록)
        requirements = "flask\npandas\nlibrosa\npyloudnorm\ntqdm\nopenpyxl"
        zf.writestr('requirements.txt', requirements)
        
    memory_file.seek(0)
    
    return send_from_directory(
        '.', # 현재 디렉토리 (의미 없음, send_file을 위해 필요)
        path=memory_file, # 실제로는 이 메모리 파일을 보냄
        as_attachment=True,
        download_name='SonaMusicAnalyzer_Project.zip',
        mimetype='application/zip'
    )
# === 추가된 부분 끝 ===

# (이하 /start_analysis, /analyze_batch, /analysis_status 경로는 이전과 동일)
@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    job_id = str(uuid.uuid4())
    analysis_progress[job_id] = {"current": 0, "total": 0, "status": "Initializing..."}
    return jsonify({"job_id": job_id})

@app.route('/analyze_batch/<job_id>', methods=['POST'])
def analyze_batch(job_id):
    files = request.files.getlist('audio_files')
    if not files or files[0].filename == '': return Response("파일이 선택되지 않았습니다.", status=400)
    total_files = len(files)
    if total_files > 100: return Response("최대 100개의 파일만 업로드할 수 있습니다.", status=400)
    analysis_progress[job_id]["total"] = total_files
    results = []
    for i, file in enumerate(files):
        analysis_progress[job_id]["current"] = i
        analysis_progress[job_id]["status"] = f"({i}/{total_files}) 분석 중: {file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        try:
            features = analyze_one(filepath)
            results.append(asdict(features))
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results.append({'file': file.filename, 'error': str(e)})
        finally:
            if os.path.exists(filepath): os.remove(filepath)
    analysis_progress[job_id]["current"] = total_files
    analysis_progress[job_id]["status"] = "분석 완료! CSV 파일 생성 중..."
    if not results:
        del analysis_progress[job_id]
        return Response("분석된 결과가 없습니다.", status=500)
    df = pd.DataFrame(results)
    csv_data = df.to_csv(index=False, encoding='utf-8-sig')
    ts = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_summary_{ts}.csv"
    del analysis_progress[job_id]
    return Response(
        csv_data,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )

@app.route('/analysis_status/<job_id>', methods=['GET'])
def get_analysis_status(job_id):
    progress = analysis_progress.get(job_id)
    if progress is None:
        return jsonify({"status": "완료 또는 찾을 수 없음", "current": 0, "total": 0}), 404
    return jsonify(progress)

if __name__ == '__main__':
    app.run(debug=True, port=5000)