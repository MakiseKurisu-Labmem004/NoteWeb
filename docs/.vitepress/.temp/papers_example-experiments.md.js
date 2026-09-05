import { resolveComponent, useSSRContext } from "vue";
import { ssrRenderAttrs, ssrRenderComponent, ssrRenderStyle } from "vue/server-renderer";
import { _ as _export_sfc } from "./plugin-vue_export-helper.1tPrXgE0.js";
const __pageData = JSON.parse('{"title":"让实验记录成为可复用的知识","description":"记录环境、参数、结果和失败原因，把一次复现变成之后能够回溯与比较的研究资料。","frontmatter":{"title":"让实验记录成为可复用的知识","description":"记录环境、参数、结果和失败原因，把一次复现变成之后能够回溯与比较的研究资料。","date":"2026-09-04","tags":["实验复现","阅读方法"],"status":"已读","venue":"阅读示例 · 实验记录","example":true},"headers":[],"relativePath":"papers/example-experiments.md","filePath":"papers/example-experiments.md","lastUpdated":null}');
const _sfc_main = { name: "papers/example-experiments.md" };
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  const _component_PaperMeta = resolveComponent("PaperMeta");
  _push(`<div${ssrRenderAttrs(_attrs)}><h1 id="让实验记录成为可复用的知识" tabindex="-1">让实验记录成为可复用的知识 <a class="header-anchor" href="#让实验记录成为可复用的知识" aria-label="Permalink to &quot;让实验记录成为可复用的知识&quot;">​</a></h1>`);
  _push(ssrRenderComponent(_component_PaperMeta, null, null, _parent));
  _push(`<div class="info custom-block"><p class="custom-block-title">示例说明</p><p>本文演示实验笔记的组织方式，不对应真实论文或已完成的实验。</p></div><h2 id="一句话总结" tabindex="-1">一句话总结 <a class="header-anchor" href="#一句话总结" aria-label="Permalink to &quot;一句话总结&quot;">​</a></h2><p>实验记录应让未来的自己知道：运行了什么、为什么运行，以及结果是否支持最初的假设。</p><h2 id="复现前检查" tabindex="-1">复现前检查 <a class="header-anchor" href="#复现前检查" aria-label="Permalink to &quot;复现前检查&quot;">​</a></h2><ul><li>[ ] 记录论文版本、代码版本与依赖版本</li><li>[ ] 确认数据预处理与划分方式</li><li>[ ] 写出预期结果和评测指标</li><li>[ ] 先运行小规模检查</li></ul><h2 id="实验日志" tabindex="-1">实验日志 <a class="header-anchor" href="#实验日志" aria-label="Permalink to &quot;实验日志&quot;">​</a></h2><table tabindex="0"><thead><tr><th>实验编号</th><th>改动</th><th>随机种子</th><th>结果</th><th>结论</th></tr></thead><tbody><tr><td>E001</td><td>原始配置</td><td>待填写</td><td>尚未运行</td><td>待验证</td></tr><tr><td>E002</td><td>移除某模块</td><td>待填写</td><td>尚未运行</td><td>待验证</td></tr></tbody></table><h3 id="配置记录" tabindex="-1">配置记录 <a class="header-anchor" href="#配置记录" aria-label="Permalink to &quot;配置记录&quot;">​</a></h3><div class="language-yaml vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">yaml</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="${ssrRenderStyle({ "--shiki-light": "#6A737D", "--shiki-dark": "#6A737D" })}"># 示例配置，请用真实实验参数替换。</span></span>
<span class="line"><span style="${ssrRenderStyle({ "--shiki-light": "#22863A", "--shiki-dark": "#85E89D" })}">experiment</span><span style="${ssrRenderStyle({ "--shiki-light": "#24292E", "--shiki-dark": "#E1E4E8" })}">: </span><span style="${ssrRenderStyle({ "--shiki-light": "#032F62", "--shiki-dark": "#9ECBFF" })}">baseline</span></span>
<span class="line"><span style="${ssrRenderStyle({ "--shiki-light": "#22863A", "--shiki-dark": "#85E89D" })}">seed</span><span style="${ssrRenderStyle({ "--shiki-light": "#24292E", "--shiki-dark": "#E1E4E8" })}">: </span><span style="${ssrRenderStyle({ "--shiki-light": "#005CC5", "--shiki-dark": "#79B8FF" })}">42</span></span>
<span class="line"><span style="${ssrRenderStyle({ "--shiki-light": "#22863A", "--shiki-dark": "#85E89D" })}">dataset</span><span style="${ssrRenderStyle({ "--shiki-light": "#24292E", "--shiki-dark": "#E1E4E8" })}">: </span><span style="${ssrRenderStyle({ "--shiki-light": "#032F62", "--shiki-dark": "#9ECBFF" })}">your-dataset</span></span>
<span class="line"><span style="${ssrRenderStyle({ "--shiki-light": "#22863A", "--shiki-dark": "#85E89D" })}">notes</span><span style="${ssrRenderStyle({ "--shiki-light": "#24292E", "--shiki-dark": "#E1E4E8" })}">: </span><span style="${ssrRenderStyle({ "--shiki-light": "#032F62", "--shiki-dark": "#9ECBFF" })}">first sanity check</span></span></code></pre></div><h2 id="失败也是结果" tabindex="-1">失败也是结果 <a class="header-anchor" href="#失败也是结果" aria-label="Permalink to &quot;失败也是结果&quot;">​</a></h2><p>区分环境错误、实现问题和方法本身的局限。保留异常信息，并写下定位问题时检查过的假设。</p><div class="tip custom-block"><p class="custom-block-title">写给未来的自己</p><p>不要只写“效果不好”。写清楚哪个指标、在哪个任务上、相对哪个基线，以及观察到什么现象。</p></div><h2 id="与其他笔记的联系" tabindex="-1">与其他笔记的联系 <a class="header-anchor" href="#与其他笔记的联系" aria-label="Permalink to &quot;与其他笔记的联系&quot;">​</a></h2><p>通过相对链接将实验记录与<a href="./example-method.html">方法拆解</a>连接起来。</p><h2 id="下一步" tabindex="-1">下一步 <a class="header-anchor" href="#下一步" aria-label="Permalink to &quot;下一步&quot;">​</a></h2><p>替换示例表格，添加自己的复现记录与结论。</p></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("papers/example-experiments.md");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const exampleExperiments = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender]]);
export {
  __pageData,
  exampleExperiments as default
};
