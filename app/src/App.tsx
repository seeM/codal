import React, { useState } from 'react';
import logo from './logo.svg';
import './App.css';

interface Message {
  text: string;
  fromUser: boolean;
}

const MessageView = (message: Message) => {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'flex-start',
      marginBottom: '10px'
    }}>
      <img src={logo} style={{ borderRadius: '50%', width: '30px', marginRight: '10px' }} alt="logo" />
      <div style={{
        backgroundColor: message.fromUser ? 'white' : 'lightgrey',
        padding: '10px',
        borderRadius: '10px',
        maxWidth: '70%',
        wordWrap: 'break-word',
        color: 'black'
      }}>
        {message.text}
      </div>
      <hr style={{ width: '100%', borderColor: 'grey' }} />
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState<string>('');

  const sendMessage = async () => {
    const message = newMessage;
    setNewMessage('');
    setMessages([...messages, { text: message, fromUser: true }]);
    const res = await fetch(`http://localhost:8000/ask/${message}`);
    const data = await res.json();
    console.log(`Response ${data}`);
    setMessages([...messages, { text: message, fromUser: true }, { text: data, fromUser: false }]);
  };

  return (
    <div className="App">
      <header className="App-header">
        {messages.map((message, index) => (
          <MessageView key={index} text={message.text} fromUser={message.fromUser} />
        ))}
      </header>
      <footer style={{
        position: 'fixed',
        bottom: 0,
        width: '100%',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        padding: '10px',
        backgroundColor: '#f0f0f0'
      }}>
        <textarea 
          value={newMessage} 
          onChange={e => setNewMessage(e.target.value)} 
          style={{
            width: '800px',
            height: '50px',
            resize: 'vertical'
          }} />
        <button onClick={sendMessage} style={{
          backgroundColor: 'blue',
          color: 'white',
          marginLeft: '10px',
          padding: '10px',
          border: 'none',
          borderRadius: '50%'
        }}>
          📤
        </button>
      </footer>
    </div>
  );
}

export default App;
