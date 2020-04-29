const drawCanvas = document.getElementById("draw");
const viewCanvas = document.getElementById("view");

const drawContext = drawCanvas.getContext("2d");
drawContext.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
const viewContext = viewCanvas.getContext("2d");
viewContext.fillRect(0, 0, drawCanvas.width, drawCanvas.height);

const judgeButton = document.getElementById("judge");
const clearButton = document.getElementById("clear");

const resultArea = document.getElementById("result");

const updateResult = (classification) => {
    let str = "";
    for(let i = 0; i <= 9; i++){
        str += `${i}: ${classification[i]}%<br>`;
    }
    resultArea.innerHTML = str;
};

judgeButton.addEventListener("click", () =>{
    viewContext.drawImage(drawCanvas, 0, 0, viewCanvas.width, viewCanvas.height);
    const data = viewContext.getImageData(0, 0, viewCanvas.width, viewCanvas.height).data;
    params = {
        img: Base64.encode(data),
        width: viewCanvas.width,
        height: viewCanvas.height,
    }
    HttpRequest.post("/predict", params, (res) => {
        updateResult(JSON.parse(res.response));
    });
});

clearButton.addEventListener("click", () =>{
    drawContext.fillStyle = "black";
    drawContext.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    viewContext.fillStyle = "black";
    viewContext.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    result.innerHTML = "";
});

let mouseDown = false;

window.addEventListener("mousedown", e =>{
    mouseDown = true;
});

window.addEventListener("mouseup", e =>{
    mouseDown = false;
});

drawCanvas.addEventListener("mousemove", e =>{
    if(mouseDown){
        let rect = e.target.getBoundingClientRect();
        let x = e.clientX - 10 - rect.left;
        let y = e.clientY - 10 - rect.top;
        drawContext.fillStyle = "white";
        drawContext.fillRect(x, y, 20, 20);
    }
});
