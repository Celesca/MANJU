'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useTableContext } from './TableContext.js';
import { TableHeadContext } from './TableHeadContext.js';
import { tableTheme } from './theme.js';

const TableHead = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = useTableContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [tableTheme.head, provider.theme?.table?.head, rootTheme?.head, props.theme],
    [get(provider.clearTheme, "table.head"), get(rootClearTheme, "head"), props.clearTheme],
    [get(provider.applyTheme, "table.head"), get(rootApplyTheme, "head"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.tableHead);
  return /* @__PURE__ */ jsx(
    TableHeadContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme },
      children: /* @__PURE__ */ jsx("thead", { ref, className: twMerge(theme.base, className), ...restProps })
    }
  );
});
TableHead.displayName = "TableHead";

export { TableHead };
//# sourceMappingURL=TableHead.js.map
