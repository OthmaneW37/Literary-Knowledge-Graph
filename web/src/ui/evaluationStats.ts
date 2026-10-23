import type { EvalResult, EvalRun } from '../api'

export function verdictMark(result:EvalResult|null|undefined) { return !result?'–':result.passed?'✓':'✗' }

/** Passed-scenario difference between the two latest complete runs of the same set. */
export function runDelta(runs:EvalRun[]) {
  const [latest,previous]=runs.filter(run=>run.state==='done')
  if(!latest||!previous||latest.scenario_count!==previous.scenario_count)return null
  return latest.summary.passed-previous.summary.passed
}

export function ago(iso:string, now=Date.now()) {
  const minutes=Math.round((now-new Date(iso).getTime())/60000)
  if(minutes<1)return 'à l’instant'
  if(minutes<60)return `il y a ${minutes} min`
  if(minutes<1440)return `il y a ${Math.round(minutes/60)} h`
  return `le ${new Date(iso).toLocaleDateString('fr-FR')}`
}
