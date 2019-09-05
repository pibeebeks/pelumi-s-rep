
function Validate(){
    var name2 = document.getElementById('name1');
    var email1 = document.getElementById('email');
    var title1 = document.getElementById('title');
    var message1 = document.getElementById('message');
    
    
    if(name2.value.trim() == ""){
        alert("Please enter your name");
        name2.focus();
        return false;
    }
    else if(name2.value.length < 4){
        alert("Please your name must be at least 4 characters");
        return false;
    }
    else if(email1.value.trim() == ""){
        alert("Please enter your email");
        email1.focus();
        return false;
    }
    else if(title1.value.trim() == ""){
        alert("Please enter a title");
        title1.focus();
        return false;
    }

    else if(message1.value.trim() == ""){
        alert("Please provide a message");
        message1.focus();
        return false;
    }
                
    else if(message1.value.length < 20){
        alert("Please  your message must be at least 20 characters");
                return false;
    }
    else{
        return true;
    }

    }

