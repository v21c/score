const { spawn } = require('child_process');

const pythonProcess = spawn('python', ['./score_model/use_model.py']);

// 질문과 답변 데이터
const data = {
    question: "당신의 강점은 무엇인가요?",
    answer: "저의 강점은 문제 해결 능력과 팀워크입니다. 어려운 상황에서도 침착하게 대처하며, 동료들과 협력하여 최선의 결과를 도출해냅니다."
};

// Python 프로세스에 데이터 전송
pythonProcess.stdin.write(JSON.stringify(data) + '\n');
pythonProcess.stdin.end();

let score;

// Python 프로세스의 표준 출력(stdout) 데이터를 받아오기
pythonProcess.stdout.on('data', (data) => {
    try {
        const result = JSON.parse(data);
        score = result.final_score;
        console.log(score);
    } catch (error) {
        console.error('결과 파싱 오류:', error);
    }
});
