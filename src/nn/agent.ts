// The AI-Agent page: everything that CAN be real computation, is.
//  - memory embeddings: char-bigram hashing → 24-d vectors (real)
//  - long-term memory clusters: k-means over those vectors (real)
//  - retrieval: cosine similarity query→memories, top-3 (real)
//  - routing: a small MLP trained at boot to pick the sub-agent (real,
//    trained on zh+en+ja task templates — the 13th net of the site)
//  - memory write-back: embed the new memory, assign to nearest centroid (real)
// The dialogue/code/terminal text per scenario is illustrative (canned).

import { Rng, mulberry32 } from './rng'
import { MLPModel, TrainSample, createMLP, forwardMLP, trainMLP } from './mlp'

export const AGENT_EMBED_DIM = 96
export const AGENT_CLUSTERS = 4
export const AGENT_ROUTES = ['retrieve', 'code', 'run', 'summarize'] as const
export type AgentLang = 'zh' | 'en' | 'ja'

/** Deterministic char uni+bigram hashing embedding, L2-normalized.
 *  (Unigrams matter for CJK, bigrams for latin words — a crude but real
 *  stand-in for the learned embeddings production agents use.) */
export function embedText(text: string): number[] {
  const v = Array.from({ length: AGENT_EMBED_DIM }, () => 0)
  const hash = (str: string, salt: number) => {
    let h = 2166136261 ^ salt
    for (const ch of str) {
      h ^= ch.codePointAt(0) ?? 0
      h = Math.imul(h, 16777619)
    }
    return Math.abs(h) % AGENT_EMBED_DIM
  }
  const s = text.toLowerCase()
  // latin/digit words as whole tokens; CJK runs as unigrams + bigrams
  for (const word of s.match(/[a-z0-9.$#_-]+/g) ?? []) v[hash(word, 17)] += 1.6
  const cjk = s.replace(/[^⺀-鿿぀-ヿ]/g, '')
  for (let i = 0; i < cjk.length; i++) {
    v[hash(cjk[i], 7)] += 1
    if (i < cjk.length - 1) v[hash(cjk.slice(i, i + 2), 131)] += 1.2
  }
  const norm = Math.sqrt(v.reduce((acc, x) => acc + x * x, 0)) || 1
  return v.map((x) => x / norm)
}

export function cosine(a: number[], b: number[]): number {
  return a.reduce((s, x, i) => s + x * b[i], 0)
}

/** Plain k-means (deterministic seed); returns assignment + centroids. */
export function kmeans(vs: number[][], k: number, rng: Rng): { assign: number[]; centroids: number[][] } {
  const d = vs[0].length
  const centroids = Array.from({ length: k }, (_, c) => [...vs[Math.floor((c * vs.length) / k)]])
  let assign = vs.map(() => 0)
  for (let it = 0; it < 24; it++) {
    assign = vs.map((v) => {
      let best = 0
      let bd = -Infinity
      centroids.forEach((c, ci) => {
        const s = cosine(v, c)
        if (s > bd) {
          bd = s
          best = ci
        }
      })
      return best
    })
    for (let c = 0; c < k; c++) {
      const members = vs.filter((_, i) => assign[i] === c)
      if (!members.length) {
        centroids[c] = [...vs[Math.floor(rng() * vs.length)]]
        continue
      }
      centroids[c] = Array.from({ length: d }, (_, j) => members.reduce((s, m) => s + m[j], 0) / members.length)
      const n = Math.sqrt(centroids[c].reduce((s, x) => s + x * x, 0)) || 1
      centroids[c] = centroids[c].map((x) => x / n)
    }
  }
  return { assign, centroids }
}

/** Top-2 PCA via power iteration → 2-d positions for the memory cloud. */
export function pca2(vs: number[][], rng: Rng): [number, number][] {
  const d = vs[0].length
  const mean = Array.from({ length: d }, (_, j) => vs.reduce((s, v) => s + v[j], 0) / vs.length)
  const X = vs.map((v) => v.map((x, j) => x - mean[j]))
  const powerIter = (deflate?: number[]): number[] => {
    let w = Array.from({ length: d }, () => rng() - 0.5)
    for (let it = 0; it < 40; it++) {
      if (deflate) {
        const proj = w.reduce((s, x, j) => s + x * deflate[j], 0)
        w = w.map((x, j) => x - proj * deflate[j])
      }
      const nw = Array.from({ length: d }, () => 0)
      for (const row of X) {
        const p = row.reduce((s, x, j) => s + x * w[j], 0)
        for (let j = 0; j < d; j++) nw[j] += p * row[j]
      }
      const n = Math.sqrt(nw.reduce((s, x) => s + x * x, 0)) || 1
      w = nw.map((x) => x / n)
    }
    return w
  }
  const p1 = powerIter()
  const p2 = powerIter(p1)
  return X.map((row) => [
    row.reduce((s, x, j) => s + x * p1[j], 0),
    row.reduce((s, x, j) => s + x * p2[j], 0),
  ])
}

// ---------------------------------------------------------------- memories

/** 16 long-term memories in 4 natural themes (project / prefs / schedule / howto). */
export const AGENT_MEMORIES: Record<AgentLang, string[]> = {
  zh: [
    '项目 deadline 是 8 月 15 日',
    '昨天写了排序函数 quickSort',
    'API 密钥存放在 .env 文件里',
    '部署要先跑 npm run build',
    '用户喜欢的咖啡口味是拿铁,不加糖',
    '用户偏好深色主题界面',
    '用户要求用中文回复',
    '用户习惯 vim 快捷键',
    '周三下午三点有项目会议',
    '明天上午十点看牙医',
    '周五之前要交周报',
    '每天早上九点站会',
    '虚拟环境用 python -m venv 创建',
    'git rebase 前要先 fetch',
    '测试服务器 IP 是 10.0.2.15',
    '重启服务用 docker restart api',
  ],
  en: [
    'Project deadline is August 15',
    'Wrote the quickSort function yesterday',
    'API keys live in the .env file',
    'Deploys must run npm run build first',
    'Favorite coffee: latte, no sugar',
    'User prefers the dark theme',
    'User wants replies in Chinese',
    'User is used to vim keybindings',
    'Project meeting Wednesday 3 pm',
    'Dentist appointment tomorrow 10 am',
    'Weekly report due before Friday',
    'Daily stand-up at 9 am',
    'Create venvs with python -m venv',
    'Always fetch before git rebase',
    'Test server IP is 10.0.2.15',
    'Restart the service with docker restart api',
  ],
  ja: [
    'プロジェクトの締切は 8 月 15 日',
    '昨日ソート関数 quickSort を書いた',
    'API キーは .env ファイルにある',
    'デプロイ前に npm run build を実行',
    'ユーザーの好きなコーヒーはラテ、砂糖なし',
    'ユーザーはダークテーマを好む',
    'ユーザーは中国語での返信を希望',
    'ユーザーは vim キーバインドに慣れている',
    '水曜 15 時にプロジェクト会議',
    '明日 10 時に歯医者',
    '金曜までに週報を提出',
    '毎朝 9 時にスタンドアップ',
    '仮想環境は python -m venv で作成',
    'git rebase の前に fetch する',
    'テストサーバーの IP は 10.0.2.15',
    'サービス再起動は docker restart api',
  ],
}

// ---------------------------------------------------------------- scenarios

export interface AgentScenario {
  /** the user's request (router input + retrieval query) */
  user: string
  plan: string
  code?: string
  command: string
  output: string
  answer: string
  /** the distilled memory written back to the vector store */
  newMemory: string
}

export const AGENT_SCENARIOS: Record<AgentLang, AgentScenario[]> = {
  zh: [
    {
      user: '把昨天写的排序函数改成降序,然后跑一遍测试',
      plan: '1. 检索记忆:找到 quickSort 函数上下文\n2. 编码 agent:改比较符为降序\n3. 执行 agent:运行测试',
      code: 'function quickSort(a) {\n  // b - a: 降序\n  return a.sort((x, y) => y - x)\n}',
      command: '$ npm test sort.spec.js',
      output: '✓ sorts descending (3ms)\n1 passing',
      answer: '已改为降序并通过测试(1 passing)。',
      newMemory: 'quickSort 已改为降序,测试通过',
    },
    {
      user: '我上次说过喜欢什么口味的咖啡?',
      plan: '1. 检索 agent:在「偏好」记忆簇中查询\n2. 直接回答,无需工具',
      command: '(无需执行命令)',
      output: '检索命中:拿铁,不加糖 (相似度见连线)',
      answer: '你喜欢拿铁,不加糖。',
      newMemory: '用户再次确认咖啡偏好:拿铁不加糖',
    },
    {
      user: '项目 deadline 是哪天?帮我设一条提醒命令',
      plan: '1. 检索 agent:项目簇 → deadline 8 月 15 日\n2. 执行 agent:写入系统提醒',
      command: '$ remind add "项目截止" 08-15 09:00',
      output: 'Reminder #42 created: 08-15 09:00',
      answer: 'deadline 是 8 月 15 日,已设置当天 9 点提醒。',
      newMemory: '已为 8 月 15 日 deadline 设置提醒 #42',
    },
    {
      user: '总结这周的工作,把要点记下来',
      plan: '1. 检索 agent:汇总本周相关记忆\n2. 记忆汇总:压缩成一条长期记忆',
      command: '(记忆压缩,无外部命令)',
      output: '5 条相关记忆 → 压缩为 1 条摘要',
      answer: '本周完成排序函数与测试,周五前交周报,deadline 8/15。',
      newMemory: '本周:完成 quickSort 降序与测试;周报周五交',
    },
  ],
  en: [
    {
      user: 'Change yesterday’s sort function to descending and run the tests',
      plan: '1. Retrieve: find the quickSort context\n2. Code agent: flip the comparator\n3. Run agent: execute tests',
      code: 'function quickSort(a) {\n  // y - x: descending\n  return a.sort((x, y) => y - x)\n}',
      command: '$ npm test sort.spec.js',
      output: '✓ sorts descending (3ms)\n1 passing',
      answer: 'Changed to descending; tests pass (1 passing).',
      newMemory: 'quickSort now descending, tests green',
    },
    {
      user: 'What is my favorite coffee again?',
      plan: '1. Retrieve agent: query the preferences cluster\n2. Answer directly, no tools needed',
      command: '(no command needed)',
      output: 'Hit: latte, no sugar (cosine scores on the lines)',
      answer: 'You like latte, no sugar.',
      newMemory: 'User re-confirmed coffee preference: latte, no sugar',
    },
    {
      user: 'When is the project deadline? Set a reminder command for it',
      plan: '1. Retrieve: project cluster → deadline Aug 15\n2. Run agent: write a system reminder',
      command: '$ remind add "project due" 08-15 09:00',
      output: 'Reminder #42 created: 08-15 09:00',
      answer: 'Deadline is Aug 15; reminder set for 9 am that day.',
      newMemory: 'Reminder #42 set for the Aug 15 deadline',
    },
    {
      user: 'Summarize this week’s work and remember the key points',
      plan: '1. Retrieve: collect this week’s memories\n2. Summarizer: compress into one long-term memory',
      command: '(memory compression, no external command)',
      output: '5 related memories → compressed to 1 summary',
      answer: 'This week: sort function done + tests; weekly report due Friday; deadline 8/15.',
      newMemory: 'Week recap: quickSort descending done; report due Friday',
    },
  ],
  ja: [
    {
      user: '昨日のソート関数を降順に変えて、テストを実行して',
      plan: '1. 検索:quickSort の文脈を取得\n2. コーディング agent:比較子を反転\n3. 実行 agent:テストを実行',
      code: 'function quickSort(a) {\n  // y - x: 降順\n  return a.sort((x, y) => y - x)\n}',
      command: '$ npm test sort.spec.js',
      output: '✓ sorts descending (3ms)\n1 passing',
      answer: '降順に変更し、テストは通過(1 passing)。',
      newMemory: 'quickSort を降順化、テスト通過',
    },
    {
      user: '前に好きだと言ったコーヒーは何だっけ?',
      plan: '1. 検索 agent:好みクラスタを照会\n2. ツール不要、直接回答',
      command: '(コマンド不要)',
      output: 'ヒット:ラテ、砂糖なし(類似度は線上に)',
      answer: 'ラテ、砂糖なしがお好みです。',
      newMemory: 'コーヒーの好みを再確認:ラテ砂糖なし',
    },
    {
      user: 'プロジェクトの締切はいつ?リマインダーを設定して',
      plan: '1. 検索:プロジェクト簇 → 締切 8/15\n2. 実行 agent:システムリマインダーを書き込む',
      command: '$ remind add "締切" 08-15 09:00',
      output: 'Reminder #42 created: 08-15 09:00',
      answer: '締切は 8 月 15 日。当日 9 時のリマインダーを設定済み。',
      newMemory: '8/15 締切のリマインダー #42 を設定',
    },
    {
      user: '今週の仕事をまとめて、要点を覚えておいて',
      plan: '1. 検索:今週の関連記憶を収集\n2. 記憶要約:1 件の長期記憶へ圧縮',
      command: '(記憶圧縮、外部コマンドなし)',
      output: '関連記憶 5 件 → 1 件の要約に圧縮',
      answer: '今週:ソート関数完成とテスト、金曜までに週報、締切 8/15。',
      newMemory: '週まとめ:quickSort 降順完了、週報は金曜',
    },
  ],
}

// ---------------------------------------------------------------- router

const ROUTE_TEMPLATES: Record<AgentLang, string[][]> = {
  zh: [
    ['我上次说了什么', '帮我找一下之前的记录', '还记得那个会议结论吗', '查查我的偏好', '之前的 deadline 是哪天', '我说过喜欢什么', '我上次说过喜欢什么口味的咖啡', '项目 deadline 是哪天'],
    ['写一个函数', '帮我改这段代码', '实现一个排序算法', '修复这个 bug', '重构这个模块', '把函数改成降序', '把昨天写的排序函数改成降序', '改成降序然后跑测试'],
    ['运行测试', '执行这个命令', '部署到服务器', '跑一遍脚本', '重启服务', '帮我设一条提醒命令'],
    ['总结这周的工作', '把要点记下来', '整理一下这些内容', '归纳会议纪要', '压缩成一句话记住', '汇总所有进展'],
  ],
  en: [
    ['what did I say before', 'find my earlier note', 'do you remember the meeting', 'look up my preference', 'when was the deadline again', 'what coffee do I like', 'what coffee did I say I like', 'when is the project deadline'],
    ['write a function', 'fix this code', 'implement a sorting algorithm', 'refactor this module', 'change the function to descending', 'debug this', 'change the sort function to descending', 'change it and run the tests'],
    ['run the tests', 'execute this command', 'deploy to the server', 'run the script', 'restart the service', 'set a reminder command'],
    ['summarize this week', 'note the key points', 'organize these items', 'condense the meeting notes', 'remember this as one line', 'wrap up the progress'],
  ],
  ja: [
    ['前に何て言ったっけ', '以前のメモを探して', '会議の結論を覚えてる', '好みを調べて', '締切はいつだったか', '好きなコーヒーは何', '好きだと言ったコーヒーは何', '締切はいつ'],
    ['関数を書いて', 'このコードを直して', 'ソートを実装して', 'バグを修正して', 'リファクタして', '降順に変えて', 'ソート関数を降順に変えて', '降順に変えてテストして'],
    ['テストを実行して', 'このコマンドを実行', 'サーバーへデプロイ', 'スクリプトを走らせて', 'サービスを再起動', 'リマインダーを設定して'],
    ['今週をまとめて', '要点を記録して', '内容を整理して', '議事録を要約して', '一行で覚えておいて', '進捗を集約して', '今週の仕事をまとめて要点を覚えて'],
  ],
}

export interface AgentTask {
  router: MLPModel
  /** router accuracy on held-out templates */
  accuracy: number
}

export function buildAgentTask(): AgentTask {
  const rng = mulberry32(0xa9e47)
  const router = createMLP(rng, [AGENT_EMBED_DIM, 16, AGENT_ROUTES.length], ['relu', 'softmax'])
  const data: TrainSample[] = []
  for (const lang of ['zh', 'en', 'ja'] as AgentLang[]) {
    ROUTE_TEMPLATES[lang].forEach((examples, cls) => {
      for (const ex of examples) {
        data.push({
          x: embedText(ex),
          y: Array.from({ length: AGENT_ROUTES.length }, (_, k) => (k === cls ? 1 : 0)),
        })
      }
    })
  }
  trainMLP(router, data, { lr: 0.08, epochs: 260, rng })
  let ok = 0
  for (const s of data) {
    const out = forwardMLP(router, s.x)
    const p = out[out.length - 1].a
    if (p.indexOf(Math.max(...p)) === s.y.indexOf(1)) ok++
  }
  return { router, accuracy: ok / data.length }
}

// ---------------------------------------------------------------- per-language trace

export interface AgentTrace {
  scenario: number
  /** router softmax over AGENT_ROUTES for the user request (real) */
  route: number[]
  /** memory cloud (real embeddings, clusters, pca positions) */
  memories: string[]
  clusters: number[]
  positions: [number, number][]
  /** retrieval: top-3 memory indices + cosine scores (real) */
  top: { idx: number; score: number }[]
  /** write-back: the new memory and its (real) nearest cluster */
  newMemory: string
  newCluster: number
  newPosition: [number, number]
}

export function buildAgentTrace(task: AgentTask, lang: AgentLang, scenario: number): AgentTrace {
  const rng = mulberry32(0xa9e47 + scenario)
  const memories = AGENT_MEMORIES[lang]
  const vecs = memories.map(embedText)
  const { assign, centroids } = kmeans(vecs, AGENT_CLUSTERS, rng)
  const pos = pca2(vecs, rng)
  const sc = AGENT_SCENARIOS[lang][scenario]
  const q = embedText(sc.user)
  const routeOut = forwardMLP(task.router, q)
  const route = routeOut[routeOut.length - 1].a
  const scored = vecs.map((v, i) => ({ idx: i, score: cosine(q, v) }))
  scored.sort((a, b) => b.score - a.score)
  const nv = embedText(sc.newMemory)
  let newCluster = 0
  let best = -Infinity
  centroids.forEach((c, ci) => {
    const s = cosine(nv, c)
    if (s > best) {
      best = s
      newCluster = ci
    }
  })
  // position the new memory near its cluster's members (mean of their pca coords)
  const members = pos.filter((_, i) => assign[i] === newCluster)
  const mx = members.reduce((s, p) => s + p[0], 0) / Math.max(1, members.length)
  const my = members.reduce((s, p) => s + p[1], 0) / Math.max(1, members.length)
  return {
    scenario,
    route,
    memories,
    clusters: assign,
    positions: pos,
    top: scored.slice(0, 3),
    newMemory: sc.newMemory,
    newCluster,
    newPosition: [mx + 0.12, my - 0.1],
  }
}
