const text = "AI powered hate speech detection system";

let index = 0;

function typeEffect(){

if(index < text.length){

document.getElementById("ai-typing").innerHTML += text.charAt(index);

index++;

setTimeout(typeEffect,35);

}

}

window.onload = typeEffect;