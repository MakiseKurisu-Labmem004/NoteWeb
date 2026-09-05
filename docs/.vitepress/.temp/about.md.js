import { ssrRenderAttrs } from "vue/server-renderer";
import { useSSRContext } from "vue";
import { _ as _export_sfc } from "./plugin-vue_export-helper.1tPrXgE0.js";
const __pageData = JSON.parse('{"title":"关于这本研究手记","description":"","frontmatter":{},"headers":[],"relativePath":"about.md","filePath":"about.md","lastUpdated":null}');
const _sfc_main = { name: "about.md" };
function _sfc_ssrRender(_ctx, _push, _parent, _attrs, $props, $setup, $data, $options) {
  _push(`<div${ssrRenderAttrs(_attrs)}><h1 id="关于这本研究手记" tabindex="-1">关于这本研究手记 <a class="header-anchor" href="#关于这本研究手记" aria-label="Permalink to &quot;关于这本研究手记&quot;">​</a></h1><p>这里是一份持续更新的论文阅读记录。</p><p>我希望每读一篇论文，都能留下三个问题的答案：</p><ol><li>它试图解决什么问题？</li><li>它为什么这样做？</li><li>它与我正在研究的问题有什么联系？</li></ol><p>比起复述摘要，我更重视对假设、实验与局限的理解。笔记会随着新的阅读和实验不断修订。</p><div class="tip custom-block"><p class="custom-block-title">自定义这页</p><p>把这段文字替换为你的研究方向、自我介绍，以及你希望公开的联系方式。</p></div><p>本站以 Markdown 编写，通过 GitHub Pages 发布。首页中标为“示例笔记”的内容仅用于展示写作结构，不代表已经完成的真实论文阅读记录。</p></div>`);
}
const _sfc_setup = _sfc_main.setup;
_sfc_main.setup = (props, ctx) => {
  const ssrContext = useSSRContext();
  (ssrContext.modules || (ssrContext.modules = /* @__PURE__ */ new Set())).add("about.md");
  return _sfc_setup ? _sfc_setup(props, ctx) : void 0;
};
const about = /* @__PURE__ */ _export_sfc(_sfc_main, [["ssrRender", _sfc_ssrRender]]);
export {
  __pageData,
  about as default
};
