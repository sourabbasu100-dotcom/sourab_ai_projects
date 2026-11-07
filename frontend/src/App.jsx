import { useRef, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function errorMessage(data, fallback) {
  if (typeof data?.detail === "string") return data.detail;
  return fallback;
}

function AppHeader() {
  return (
    <header className="topbar">
      <a className="brand" href="/" aria-label="PDF RAG Chat home">
        <span className="brand-mark">R</span>
        <span>paper<span className="brand-light">chat</span></span>
      </a>
      <div className="connection"><span /> Local and private</div>
    </header>
  );
}

function ChatHeading() {
  return (
    <div className="chat-heading">
      <div className="eyebrow">YOUR DOCUMENT ASSISTANT</div>
      <h1>Ask your documents.</h1>
      <p>Upload a PDF, then get answers grounded in its contents.</p>
    </div>
  );
}

function DocumentUpload({ fileInput, uploading, onUpload }) {
  return (
    <div className="document-bar">
      <div className="document-icon" aria-hidden="true">PDF</div>
      <div className="document-copy">
        <strong>Knowledge base</strong>
        <span>Upload a PDF to get started</span>
      </div>
      <input
        ref={fileInput}
        className="visually-hidden"
        type="file"
        accept="application/pdf,.pdf"
        onChange={onUpload}
        aria-label="Choose a PDF to upload"
      />
      <button
        className="upload-button"
        type="button"
        onClick={() => fileInput.current?.click()}
        disabled={uploading}
      >
        {uploading ? "Adding..." : "＋ Add PDF"}
      </button>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <div className="sparkle" aria-hidden="true">✳</div>
      <strong>Your next great question starts here.</strong>
      <span>Try asking for a summary, a key detail, or an explanation.</span>
    </div>
  );
}

function ChatMessage({ message }) {
  const label =
    message.role === "user" ? "YOU" : message.role === "error" ? "NOTICE" : "PAPERCHAT";

  return (
    <article className={`message ${message.role}`}>
      <div className="message-label">{label}</div>
      <div className="message-text">{message.text}</div>
      {message.sources?.length > 0 && (
        <div className="sources">
          <span>Sources</span>
          {message.sources.map((source, index) => (
            <span className="source-chip" key={`${source.source}-${source.page}-${index}`}>
              {source.source.split(/[\\/]/).pop()}, p. {source.page}
            </span>
          ))}
        </div>
      )}
    </article>
  );
}

function TypingIndicator() {
  return (
    <div className="message assistant">
      <div className="message-label">PAPERCHAT</div>
      <div className="thinking"><span /><span /><span /> Searching your documents...</div>
    </div>
  );
}

function MessageList({ messages, sending }) {
  return (
    <div className="messages" aria-live="polite">
      {messages.length === 0 ? (
        <EmptyState />
      ) : (
        messages.map((message, index) => (
          <ChatMessage key={`${message.role}-${index}`} message={message} />
        ))
      )}
      {sending && <TypingIndicator />}
    </div>
  );
}

function ChatComposer({ question, sending, onQuestionChange, onSubmit }) {
  return (
    <form className="composer" onSubmit={onSubmit}>
      <input
        value={question}
        onChange={(event) => onQuestionChange(event.target.value)}
        placeholder="Ask a question about your PDF..."
        aria-label="Your question"
        disabled={sending}
      />
      <button type="submit" disabled={!question.trim() || sending} aria-label="Send question">
        <span>Ask</span><span aria-hidden="true">↗</span>
      </button>
    </form>
  );
}

function PrivacyNote() {
  return (
    <div className="privacy-note">
      Answers are generated locally with your documents and Ollama.
    </div>
  );
}

function ChatPanel({
  fileInput,
  messages,
  notice,
  onQuestionChange,
  onSubmit,
  onUpload,
  question,
  sending,
  uploading,
}) {
  return (
    <section className="chat-panel" aria-label="PDF chatbot">
      <ChatHeading />
      <DocumentUpload fileInput={fileInput} uploading={uploading} onUpload={onUpload} />
      <MessageList messages={messages} sending={sending} />
      {notice && <div className="notice" role="status">{notice}</div>}
      <ChatComposer
        question={question}
        sending={sending}
        onQuestionChange={onQuestionChange}
        onSubmit={onSubmit}
      />
      <PrivacyNote />
    </section>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [notice, setNotice] = useState("");
  const fileInput = useRef(null);

  async function uploadPdf(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setNotice("");
    setUploading(true);
    const body = new FormData();
    body.append("file", file);

    try {
      const response = await fetch(`${API_URL}/api/documents`, {
        method: "POST",
        body,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(errorMessage(data, "Could not ingest PDF."));
      setNotice(`Added ${data.filename} (${data.chunks} text chunks).`);
    } catch (error) {
      setNotice(error.message || "Could not connect to the RAG API.");
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  async function sendQuestion(event) {
    event.preventDefault();
    const text = question.trim();
    if (!text || sending) return;

    setMessages((current) => [...current, { role: "user", text }]);
    setQuestion("");
    setNotice("");
    setSending(true);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(errorMessage(data, "Could not get an answer."));
      setMessages((current) => [
        ...current,
        { role: "assistant", text: data.answer, sources: data.sources },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "error", text: error.message || "Could not connect to the RAG API." },
      ]);
    } finally {
      setSending(false);
    }
  }

  return (
    <main className="app-shell">
      <AppHeader />
      <ChatPanel
        fileInput={fileInput}
        messages={messages}
        notice={notice}
        onQuestionChange={setQuestion}
        onSubmit={sendQuestion}
        onUpload={uploadPdf}
        question={question}
        sending={sending}
        uploading={uploading}
      />
    </main>
  );
}
