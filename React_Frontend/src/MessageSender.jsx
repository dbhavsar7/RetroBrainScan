import { useState } from "react";

export default function MessageSender() {
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState("");

  const sendMessage = async () => {
    try {
      const res = await fetch("http://127.0.0.1:5000/message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      const text = await res.text();   // because Flask returns plain text
      setResponse(text);
    } catch (error) {
      console.error(error);
      setResponse("Error sending message");
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Flask + React Test</h1>

      <input
        type="text"
        placeholder="Type a message..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        style={{ padding: 8, marginRight: 10 }}
      />

      <button onClick={sendMessage} style={{ padding: "8px 12px" }}>
        Send
      </button>

      <h3>Response from Flask:</h3>
      <p>{response}</p>
    </div>
  );
}
