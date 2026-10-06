import {useEffect,useState,type ImgHTMLAttributes} from 'react'

function useMedia(src:string){
  const [url,setUrl]=useState('')
  useEffect(()=>{
    if(!src.startsWith('/api/visuals/')){setUrl(src);return}
    const controller=new AbortController();let objectUrl=''
    setUrl('')
    const token=localStorage.getItem('narrativelens:token')
    fetch(src,{signal:controller.signal,headers:token?{Authorization:`Bearer ${token}`}:{}})
      .then(response=>{if(!response.ok)throw new Error('Image inaccessible');return response.blob()})
      .then(blob=>{if(!controller.signal.aborted){objectUrl=URL.createObjectURL(blob);setUrl(objectUrl)}})
      .catch(()=>{})
    return()=>{controller.abort();if(objectUrl)URL.revokeObjectURL(objectUrl)}
  },[src])
  return url
}
export function ProtectedImage({src='',...props}:ImgHTMLAttributes<HTMLImageElement>){
  const url=useMedia(src)
  return url?<img {...props} src={url}/>:<span className={props.className} aria-label="Image indisponible"/>
}
export function ProtectedVisual({src,caption}:{src:string;caption:string}){
  const url=useMedia(src)
  return url?<a href={url} target="_blank" rel="noreferrer"><img src={url} alt={caption}/><span>{caption}</span></a>:<span>Image inaccessible</span>
}
