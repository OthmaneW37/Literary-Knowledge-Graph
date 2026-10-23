import type { Book, DiscoveredBook, OpenJob } from '../api'

const fold=(value:string)=>value.normalize('NFD').replace(/[̀-ͯ]/g,'').toLowerCase()

/** Books already in the library whose title or author contains every query word. */
export function localMatches(library:Book[], query:string) {
  const words=fold(query).split(/[^a-z0-9]+/).filter(word=>word.length>2)
  if(!words.length)return []
  return library.filter(book=>!book.archived&&words.every(word=>fold(`${book.title} ${book.author}`).includes(word)))
}

/** 0-100 progress of a book being opened: download and indexing first, then embeddings. */
export function openProgress(job:OpenJob|undefined) {
  if(!job)return 0
  if(job.state==='done')return 100
  if(job.state==='ready')return job.to_embed?40+Math.round(60*job.embedded/job.to_embed):40
  return {queued:5,downloading:15,indexing:30,failed:0}[job.state]
}

/** Key shared by the Explorer cards and the App's open jobs. */
export function jobKey(book:Pick<DiscoveredBook,'provider'|'provider_id'>|OpenJob) {
  return `${book.provider==='wikisource'?'wikisource':'gutenberg'}:${book.provider_id}`
}
