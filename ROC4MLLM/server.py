from fastapi import FastAPI
from mplug_owl2.assessor import Assessment
from PIL import Image

assessment=Assessment(pretrained="/home/mxy/ROC4MLLM_weights")

app = FastAPI(title="ROC4MLLM")


@app.post("/api/roc4mllm")
async def score_roc4mllm(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(content={"error": "未选择文件"}, status_code=400)

    temp_file_path = None  # 初始化变量
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            # 读取上传文件内容并写入临时文件
            contents = await file.read()
            tmp_file.write(contents)
            temp_file_path = tmp_file.name

        # ref_cc, model = color_eval_init()

        # 使用临时文件路径调用评估函数
        # result = eval_sbj_img_color(temp_file_path, model, threshold)

        img = Image.open(temp_file_path).convert('RGB')
        input_img=[img]
        answer,score = assessment(input_img, precision=4)

        result = {"score":score[0], "comment": answer[0]}

        # 转换为JSON字符串
        # json_result = json.dumps(result, ensure_ascii=False, indent=2)
        # print(json_result)

        # 或者直接返回字典（如果是API响应）
        return result

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # 确保删除临时文件
        if temp_file_path is not None and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)



