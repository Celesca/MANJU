'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var TableContext = require('./TableContext.cjs');
var TableHeadContext = require('./TableHeadContext.cjs');
var theme = require('./theme.cjs');

const TableHead = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = TableContext.useTableContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.tableTheme.head, provider$1.theme?.table?.head, rootTheme?.head, props.theme],
    [get.get(provider$1.clearTheme, "table.head"), get.get(rootClearTheme, "head"), props.clearTheme],
    [get.get(provider$1.applyTheme, "table.head"), get.get(rootApplyTheme, "head"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.tableHead);
  return /* @__PURE__ */ jsxRuntime.jsx(
    TableHeadContext.TableHeadContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme },
      children: /* @__PURE__ */ jsxRuntime.jsx("thead", { ref, className: tailwindMerge.twMerge(theme$1.base, className), ...restProps })
    }
  );
});
TableHead.displayName = "TableHead";

exports.TableHead = TableHead;
//# sourceMappingURL=TableHead.cjs.map
