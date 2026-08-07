import { Arch, useStore, useT } from '../store'

interface Paper {
  title: string
  authors: string
  year: number
  url: string
  /** what this paper contributes to the arch shown (i18n key suffix, optional) */
  note?: string
}

/** The canonical papers behind each architecture — language-independent. */
const PAPERS: Record<Arch, Paper[]> = {
  mlp: [
    {
      title: 'Learning Representations by Back-propagating Errors',
      authors: 'Rumelhart, Hinton & Williams',
      year: 1986,
      url: 'https://www.nature.com/articles/323533a0',
    },
    {
      title: 'The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain',
      authors: 'Rosenblatt',
      year: 1958,
      url: 'https://psycnet.apa.org/doi/10.1037/h0042519',
    },
  ],
  cnn: [
    {
      title: 'Gradient-Based Learning Applied to Document Recognition (LeNet)',
      authors: 'LeCun, Bottou, Bengio & Haffner',
      year: 1998,
      url: 'https://ieeexplore.ieee.org/document/726791',
    },
    {
      title: 'ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)',
      authors: 'Krizhevsky, Sutskever & Hinton',
      year: 2012,
      url: 'https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html',
    },
  ],
  rnn: [
    {
      title: 'Finding Structure in Time',
      authors: 'Elman',
      year: 1990,
      url: 'https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1',
    },
    {
      title: 'Learning Long-Term Dependencies with Gradient Descent is Difficult',
      authors: 'Bengio, Simard & Frasconi',
      year: 1994,
      url: 'https://ieeexplore.ieee.org/document/279181',
    },
  ],
  lstm: [
    {
      title: 'Long Short-Term Memory',
      authors: 'Hochreiter & Schmidhuber',
      year: 1997,
      url: 'https://www.bioinf.jku.at/publications/older/2604.pdf',
    },
    {
      title: 'LSTM: A Search Space Odyssey',
      authors: 'Greff, Srivastava, Koutník, Steunebrink & Schmidhuber',
      year: 2015,
      url: 'https://arxiv.org/abs/1503.04069',
    },
  ],
  llm: [
    {
      title: 'Attention Is All You Need',
      authors: 'Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin',
      year: 2017,
      url: 'https://arxiv.org/abs/1706.03762',
    },
    {
      title: 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer',
      authors: 'Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton & Dean',
      year: 2017,
      url: 'https://arxiv.org/abs/1701.06538',
    },
    {
      title: 'Language Models are Few-Shot Learners (GPT-3)',
      authors: 'Brown et al.',
      year: 2020,
      url: 'https://arxiv.org/abs/2005.14165',
    },
  ],
  ae: [
    {
      title: 'Reducing the Dimensionality of Data with Neural Networks',
      authors: 'Hinton & Salakhutdinov',
      year: 2006,
      url: 'https://www.science.org/doi/10.1126/science.1127647',
    },
    {
      title: 'Auto-Encoding Variational Bayes (VAE)',
      authors: 'Kingma & Welling',
      year: 2013,
      url: 'https://arxiv.org/abs/1312.6114',
    },
  ],
  diff: [
    {
      title: 'Denoising Diffusion Probabilistic Models (DDPM)',
      authors: 'Ho, Jain & Abbeel',
      year: 2020,
      url: 'https://arxiv.org/abs/2006.11239',
    },
    {
      title: 'Deep Unsupervised Learning using Nonequilibrium Thermodynamics',
      authors: 'Sohl-Dickstein, Weiss, Maheswaranathan & Ganguli',
      year: 2015,
      url: 'https://arxiv.org/abs/1503.03585',
    },
    {
      title: 'High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)',
      authors: 'Rombach, Blattmann, Lorenz, Esser & Ommer',
      year: 2021,
      url: 'https://arxiv.org/abs/2112.10752',
    },
  ],
  mamba: [
    {
      title: 'Mamba: Linear-Time Sequence Modeling with Selective State Spaces',
      authors: 'Gu & Dao',
      year: 2023,
      url: 'https://arxiv.org/abs/2312.00752',
    },
    {
      title: 'Efficiently Modeling Long Sequences with Structured State Spaces (S4)',
      authors: 'Gu, Goel & Ré',
      year: 2021,
      url: 'https://arxiv.org/abs/2111.00396',
    },
    {
      title: 'Transformers are SSMs: Generalized Models and Efficient Algorithms (Mamba-2)',
      authors: 'Dao & Gu',
      year: 2024,
      url: 'https://arxiv.org/abs/2405.21060',
    },
  ],
  agent: [
    {
      title: 'ReAct: Synergizing Reasoning and Acting in Language Models',
      authors: 'Yao, Zhao, Yu, Du, Shafran, Narasimhan & Cao',
      year: 2022,
      url: 'https://arxiv.org/abs/2210.03629',
    },
    {
      title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)',
      authors: 'Lewis et al.',
      year: 2020,
      url: 'https://arxiv.org/abs/2005.11401',
    },
    {
      title: 'Generative Agents: Interactive Simulacra of Human Behavior',
      authors: 'Park, O’Brien, Cai, Morris, Liang & Bernstein',
      year: 2023,
      url: 'https://arxiv.org/abs/2304.03442',
    },
    {
      title: 'Toolformer: Language Models Can Teach Themselves to Use Tools',
      authors: 'Schick et al.',
      year: 2023,
      url: 'https://arxiv.org/abs/2302.04761',
    },
  ],
  giant: [
    {
      title: 'Language Models are Few-Shot Learners (GPT-3)',
      authors: 'Brown et al.',
      year: 2020,
      url: 'https://arxiv.org/abs/2005.14165',
    },
    {
      title: 'The Llama 3 Herd of Models',
      authors: 'Meta AI',
      year: 2024,
      url: 'https://arxiv.org/abs/2407.21783',
    },
    {
      title: 'DeepSeek-V3 Technical Report',
      authors: 'DeepSeek-AI',
      year: 2024,
      url: 'https://arxiv.org/abs/2412.19437',
    },
    {
      title: 'Scaling Laws for Neural Language Models',
      authors: 'Kaplan, McCandlish et al.',
      year: 2020,
      url: 'https://arxiv.org/abs/2001.08361',
    },
  ],
  vit: [
    {
      title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)',
      authors: 'Dosovitskiy et al.',
      year: 2020,
      url: 'https://arxiv.org/abs/2010.11929',
    },
    {
      title: 'Attention Is All You Need',
      authors: 'Vaswani et al.',
      year: 2017,
      url: 'https://arxiv.org/abs/1706.03762',
    },
    {
      title: 'Learning Transferable Visual Models From Natural Language Supervision (CLIP)',
      authors: 'Radford et al.',
      year: 2021,
      url: 'https://arxiv.org/abs/2103.00020',
    },
  ],
  gnn: [
    {
      title: 'Semi-Supervised Classification with Graph Convolutional Networks (GCN)',
      authors: 'Kipf & Welling',
      year: 2016,
      url: 'https://arxiv.org/abs/1609.02907',
    },
    {
      title: 'The Graph Neural Network Model',
      authors: 'Scarselli, Gori, Tsoi, Hagenbuchner & Monfardini',
      year: 2009,
      url: 'https://ieeexplore.ieee.org/document/4700287',
    },
    {
      title: 'Graph Attention Networks (GAT)',
      authors: 'Veličković, Cucurull, Casanova, Romero, Liò & Bengio',
      year: 2017,
      url: 'https://arxiv.org/abs/1710.10903',
    },
  ],
  gan: [
    {
      title: 'Generative Adversarial Networks',
      authors: 'Goodfellow, Pouget-Abadie, Mirza, Xu, Warde-Farley, Ozair, Courville & Bengio',
      year: 2014,
      url: 'https://arxiv.org/abs/1406.2661',
    },
    {
      title: 'A Style-Based Generator Architecture for GANs (StyleGAN)',
      authors: 'Karras, Laine & Aila',
      year: 2018,
      url: 'https://arxiv.org/abs/1812.04948',
    },
  ],
  text: [
    {
      title: 'Learning Representations by Back-propagating Errors',
      authors: 'Rumelhart, Hinton & Williams',
      year: 1986,
      url: 'https://www.nature.com/articles/323533a0',
    },
    {
      title: 'N-Gram-Based Text Categorization',
      authors: 'Cavnar & Trenkle',
      year: 1994,
      url: 'https://www.let.rug.nl/vannoord/TextCat/textcat.pdf',
    },
  ],
}

/** Whole-architecture overview: what it is, what problems it solves,
 *  where it is used and which industries it powers. */
export function OverviewPanel() {
  const arch = useStore((s) => s.arch)
  const open = useStore((s) => s.overviewOpen)
  const toggleOverview = useStore((s) => s.toggleOverview)
  const t = useT()

  if (!open) return null

  const bullets = (key: string) =>
    t(key)
      .split('\n')
      .filter(Boolean)
      .map((line, i) => <li key={i}>{line}</li>)

  return (
    <aside className="inspector overview">
      <header className="inspector-head">
        <div>
          <div className="inspector-title">{t(`arch.${arch}Full`)}</div>
          <div className="inspector-sub">{t('overview.title')}</div>
        </div>
        <button className="icon-btn" onClick={toggleOverview} title={t('panel.close')}>
          ×
        </button>
      </header>
      <div className="inspector-body">
        <section className="explain-section what">
          <h4>{t('overview.secIntro')}</h4>
          <p className="explain-text">{t(`overview.${arch}.intro`)}</p>
        </section>
        <section className="explain-section why">
          <h4>{t('overview.secProblems')}</h4>
          <p className="explain-text">{t(`overview.${arch}.problems`)}</p>
        </section>
        <section className="explain-section simple">
          <h4>{t('overview.secDomains')}</h4>
          <ul className="overview-list">{bullets(`overview.${arch}.domains`)}</ul>
        </section>
        <section className="explain-section industries">
          <h4>{t('overview.secIndustries')}</h4>
          <ul className="overview-list">{bullets(`overview.${arch}.industries`)}</ul>
        </section>
        <section className="explain-section papers">
          <h4>{t('overview.secPapers')}</h4>
          <ul className="paper-list">
            {PAPERS[arch].map((p) => (
              <li key={p.url}>
                <a href={p.url} target="_blank" rel="noreferrer" className="paper-link">
                  <span className="paper-title">{p.title}</span>
                  <span className="paper-meta">
                    {p.authors} · {p.year} ↗
                  </span>
                </a>
              </li>
            ))}
          </ul>
        </section>
        <div className="explain-tip">{t('overview.tip')}</div>
      </div>
    </aside>
  )
}
