const text = "AI powered hate speech detection system"

let index = 0

function typeEffect(){

if(index < text.length){

document.getElementById("ai-typing").innerHTML += text.charAt(index)

index++

setTimeout(typeEffect,35)

}

}

typeEffect()

const btn = document.getElementById("analyzeBtn")

if(btn){

btn.addEventListener("click",()=>{

document.getElementById("spinner").classList.remove("hidden")

})

}