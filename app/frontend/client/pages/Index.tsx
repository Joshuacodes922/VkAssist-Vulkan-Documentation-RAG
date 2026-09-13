import { FormEvent, type ReactNode, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { ArrowLeft, ArrowUp, BookOpen, Code2, FileText, Plus } from "lucide-react";
import frontendReadme from "@/content/README.md?raw";
import { chatWithVkAssist, sourceDocumentUrl, type ChatMessage, type Citation } from "@/lib/vkassist";
import { cn } from "@/lib/utils";

const githubUrl = "https://github.com/Joshuacodes922/Vulkan-Assistance-Chatbot";

function CitationLinks({ citations, openCitation }: { citations?: Citation[]; openCitation: (citation: Citation) => void }) {
  if (!citations?.length) return null;
  return <div className="mt-4 flex flex-wrap gap-2">{citations.map((citation) => <button key={citation.id} onClick={() => openCitation(citation)} className="flex items-center gap-1.5 rounded-md border border-[#393939] bg-[#171717] px-2.5 py-1.5 text-left text-[11px] text-[#b8b8b8] hover:border-[#b7352b] hover:text-white"><FileText size={13} className="text-[#e34a3b]" />[{citation.id}] {citation.section || "Source"} · p. {citation.page_start}</button>)}</div>;
}

function Answer({ message, openCitation }: { message: ChatMessage; openCitation: (citation: Citation) => void }) {
  const citationMap = new Map((message.citations ?? []).map((citation) => [String(citation.id), citation]));
  return <article className={cn("max-w-3xl text-[15px] leading-7 text-[#e5e5e5]", message.error && "text-[#ff9d96]")}><div>{message.content.split(/(\[\d+\])/g).map((part, index) => {
    const citation = citationMap.get(part.slice(1, -1));
    return citation ? <button key={index} onClick={() => openCitation(citation)} className="mx-0.5 rounded bg-[#381b1a] px-1.5 py-0.5 text-xs font-semibold text-[#ff796d] hover:bg-[#6b2924] hover:text-white">[{citation.id}]</button> : <span key={index}>{part}</span>;
  })}</div><CitationLinks citations={message.citations} openCitation={openCitation} /></article>;
}

function Composer({ value, onChange, onSubmit, loading }: { value: string; onChange: (value: string) => void; onSubmit: (event: FormEvent) => void; loading: boolean }) {
  return <form onSubmit={onSubmit} className="w-full"><div className="flex w-full items-end gap-2 rounded-[1.4rem] border border-[#393939] bg-[#242424] px-3 py-2 shadow-[0_8px_30px_rgb(0,0,0,0.35)] focus-within:border-[#a83128]"><button type="button" aria-label="Add attachment" className="mb-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full text-[#a9a9a9] hover:bg-[#353535] hover:text-white"><Plus size={19} /></button><textarea value={value} onChange={(event) => onChange(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) onSubmit(event); }} placeholder="Ask VkAssist" rows={1} className="max-h-40 min-h-9 flex-1 resize-none bg-transparent py-1.5 text-[15px] leading-6 text-white outline-none placeholder:text-[#9a9a9a]" /><button disabled={!value.trim() || loading} aria-label="Send" className="mb-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full bg-[#e23b2d] text-white hover:bg-[#f15042] disabled:bg-[#515151] disabled:text-[#929292]"><ArrowUp size={17} strokeWidth={2.7} /></button></div></form>;
}

function TopActions() {
  const buttonClass = "flex items-center gap-2 rounded-md border border-[#3a3a3a] bg-[#111] px-3 py-2 text-xs text-[#c8c8c8] hover:border-[#b7352b] hover:text-white";
  return <div className="absolute right-5 top-5 z-10 flex gap-2"><button onClick={() => window.open("/readme", "_blank", "noopener,noreferrer")} className={buttonClass}><FileText size={14} /> Read docs</button><button onClick={() => window.open("/documents", "_blank", "noopener,noreferrer")} className={buttonClass}><BookOpen size={14} /> Preview source</button><a href={githubUrl} target="_blank" rel="noreferrer" className={buttonClass}><Code2 size={14} /> GitHub</a></div>;
}

function ChatPage() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const openCitation = (citation: Citation) => navigate(`/documents?page=${citation.page_start}`);
  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const question = input.trim();
    if (!question || loading) return;
    setMessages((current) => [...current, { id: `user-${Date.now()}`, role: "user", content: question }]);
    setInput(""); setLoading(true);
    try {
      const answer = await chatWithVkAssist(question);
      setMessages((current) => [...current, answer]);
    }
    catch (error) { setMessages((current) => [...current, { id: `error-${Date.now()}`, role: "assistant", content: error instanceof Error ? error.message : "Unable to generate an answer.", error: true }]); }
    finally { setLoading(false); }
  }
  const empty = messages.length === 0;
  return <main className="relative flex h-screen w-screen min-w-0 flex-col overflow-hidden bg-black text-white"><TopActions /><section className={cn("flex min-h-0 flex-1 flex-col", empty ? "justify-center pb-[13vh]" : "overflow-y-auto px-5 py-16")}><div className={cn("mx-auto w-full max-w-3xl px-5", empty && "max-w-[760px]")}>{empty ? <><div className="mb-8 text-center"><h1 className="text-2xl font-medium tracking-tight text-[#f2f2f2]">Vulkan RAG Application</h1><p className="mt-3 text-sm text-[#a5a5a5]">Grounded answers from the indexed Vulkan documentation.</p></div><Composer value={input} onChange={setInput} onSubmit={submit} loading={loading} /></> : <div className="space-y-8">{messages.map((message) => message.role === "user" ? <div key={message.id} className="ml-auto max-w-[82%] rounded-2xl bg-[#252525] px-4 py-3 text-[15px] leading-6 text-[#f2f2f2]">{message.content}</div> : <Answer key={message.id} message={message} openCitation={openCitation} />)}{loading && <div className="flex items-center gap-2 text-sm text-[#a4a4a4]"><span className="h-2 w-2 animate-pulse rounded-full bg-[#df3f31]" /> Searching the Vulkan documentation…</div>}</div>}</div></section>{!empty && <footer className="border-t border-[#202020] bg-black px-5 py-4"><div className="mx-auto max-w-3xl"><Composer value={input} onChange={setInput} onSubmit={submit} loading={loading} /></div></footer>}</main>;
}

function inlineMarkdown(value: string): ReactNode[] {
  return value.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).filter(Boolean).map((part, index) => {
    if (part.startsWith("`")) return <code key={index} className="rounded bg-[#231817] px-1.5 py-0.5 font-mono text-[0.86em] text-[#ff9a8d]">{part.slice(1, -1)}</code>;
    if (part.startsWith("**")) return <strong key={index} className="font-semibold text-white">{part.slice(2, -2)}</strong>;
    return <span key={index}>{part}</span>;
  });
}

function ReadmePage() {
  const blocks = frontendReadme.trim().split(/\n\s*\n/);
  return <main className="min-h-screen bg-black px-5 py-12 text-[#cfcfcf]"><div className="mx-auto max-w-4xl"><button onClick={() => window.close()} className="mb-12 flex items-center gap-2 text-xs text-[#999] hover:text-white"><ArrowLeft size={14} /> Close documentation</button><article className="rounded-2xl border border-[#302827] bg-[#111] px-6 py-10 shadow-[0_0_45px_rgb(180,45,35,0.10)] sm:px-12"><div className="mb-10 border-b border-[#332625] pb-8"><p className="text-xs font-semibold tracking-[.18em] text-[#dc4b3e]">PROJECT DOCUMENTATION</p><h1 className="mt-3 text-3xl font-semibold text-white">vkassist</h1><p className="mt-3 text-sm text-[#999]">A readable project guide, bundled directly with the frontend.</p></div>{blocks.map((block, index) => {
    if (block.startsWith("```")) { const code = block.replace(/^```[^\n]*\n?/, "").replace(/```$/, ""); return <pre key={index} className="my-6 overflow-x-auto rounded-xl border border-[#352827] bg-[#090909] p-4 text-xs leading-6 text-[#e7ddd8]"><code>{code}</code></pre>; }
    if (block.startsWith("# ")) return null;
    if (block.startsWith("## ")) return <h2 key={index} className="mt-12 border-b border-[#2b2524] pb-3 text-xl font-semibold text-white">{block.slice(3)}</h2>;
    if (block.startsWith("### ")) return <h3 key={index} className="mt-8 text-base font-semibold text-[#ff786b]">{block.slice(4)}</h3>;
    if (block.startsWith("- ")) return <ul key={index} className="my-5 space-y-3 pl-5 text-sm leading-6 marker:text-[#df493b]">{block.split("\n").map((line) => <li key={line}>{inlineMarkdown(line.slice(2))}</li>)}</ul>;
    if (block.startsWith("|")) return <pre key={index} className="my-5 overflow-x-auto rounded-lg border border-[#302827] bg-[#0c0c0c] p-4 text-xs leading-6 text-[#bbb]">{block}</pre>;
    return <p key={index} className="my-5 text-sm leading-7 text-[#c3c3c3]">{inlineMarkdown(block)}</p>;
  })}</article></div></main>;
}

function DocumentsPage() {
  const page = Math.max(1, Number(new URLSearchParams(useLocation().search).get("page")) || 1);
  return <main className="h-screen w-screen bg-black p-3"><div className="h-full overflow-hidden rounded-lg border border-[#a6372d] bg-[#171717] shadow-[0_0_28px_rgb(205,45,34,0.14)]"><iframe title="Indexed Vulkan documentation" src={`${sourceDocumentUrl}#page=${page}`} className="h-full min-h-[600px] w-full bg-white" /></div></main>;
}

export default function Index() {
  const path = useLocation().pathname;
  if (path === "/documents") return <DocumentsPage />;
  if (path === "/readme") return <ReadmePage />;
  return <ChatPage />;
}
