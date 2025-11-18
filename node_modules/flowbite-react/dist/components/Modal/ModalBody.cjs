'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var ModalContext = require('./ModalContext.cjs');
var theme = require('./theme.cjs');

const ModalBody = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, popup } = ModalContext.useModalContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.modalTheme.body, provider$1.theme?.modal?.body, rootTheme?.body, props.theme],
    [get.get(provider$1.clearTheme, "modal.body"), get.get(rootClearTheme, "body"), props.clearTheme],
    [get.get(provider$1.applyTheme, "modal.body"), get.get(rootApplyTheme, "body"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.modalBody);
  return /* @__PURE__ */ jsxRuntime.jsx("div", { ref, className: tailwindMerge.twMerge(theme$1.base, popup && theme$1.popup, className), ...restProps });
});
ModalBody.displayName = "ModalBody";

exports.ModalBody = ModalBody;
//# sourceMappingURL=ModalBody.cjs.map
