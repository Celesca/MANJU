'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var chevronLeftIcon = require('../../icons/chevron-left-icon.cjs');
var chevronRightIcon = require('../../icons/chevron-right-icon.cjs');
var provider = require('../../theme/provider.cjs');
var helpers = require('./helpers.cjs');
var PaginationButton = require('./PaginationButton.cjs');
var theme = require('./theme.cjs');

const Pagination = React.forwardRef((props, ref) => {
  if (props.layout === "table") return /* @__PURE__ */ jsxRuntime.jsx(TablePagination, { ...props, ref });
  return /* @__PURE__ */ jsxRuntime.jsx(DefaultPagination, { ...props, ref });
});
const DefaultPagination = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.paginationTheme, provider$1.theme?.pagination, props.theme],
    [get.get(provider$1.clearTheme, "pagination"), props.clearTheme],
    [get.get(provider$1.applyTheme, "pagination"), props.applyTheme]
  );
  const {
    className,
    currentPage,
    layout = "pagination",
    nextLabel = "Next",
    onPageChange,
    previousLabel = "Previous",
    renderPaginationButton = (props2) => /* @__PURE__ */ jsxRuntime.jsx(PaginationButton.PaginationButton, { ...props2 }),
    totalPages,
    showIcons: showIcon = false,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.pagination);
  if (!Number.isInteger(currentPage) || currentPage < 1) {
    throw new Error("Invalid props: currentPage must be a positive integer");
  }
  if (!Number.isInteger(totalPages) || totalPages < 1) {
    throw new Error("Invalid props: totalPages must be a positive integer");
  }
  const lastPage = Math.min(Math.max(layout === "pagination" ? currentPage + 2 : currentPage + 4, 5), totalPages);
  const firstPage = Math.max(1, lastPage - 4);
  function goToNextPage() {
    onPageChange(Math.min(currentPage + 1, totalPages));
  }
  function goToPreviousPage() {
    onPageChange(Math.max(currentPage - 1, 1));
  }
  return /* @__PURE__ */ jsxRuntime.jsx("nav", { ref, className: tailwindMerge.twMerge(theme$1.base, className), ...restProps, children: /* @__PURE__ */ jsxRuntime.jsxs("ul", { className: theme$1.pages.base, children: [
    /* @__PURE__ */ jsxRuntime.jsx("li", { children: /* @__PURE__ */ jsxRuntime.jsxs(
      PaginationButton.PaginationNavigation,
      {
        className: tailwindMerge.twMerge(theme$1.pages.previous.base, showIcon && theme$1.pages.showIcon),
        onClick: goToPreviousPage,
        disabled: currentPage === 1,
        children: [
          showIcon && /* @__PURE__ */ jsxRuntime.jsx(chevronLeftIcon.ChevronLeftIcon, { "aria-hidden": true, className: theme$1.pages.previous.icon }),
          previousLabel
        ]
      }
    ) }),
    layout === "pagination" && helpers.range(firstPage, lastPage).map((page) => /* @__PURE__ */ jsxRuntime.jsx("li", { "aria-current": page === currentPage ? "page" : void 0, children: renderPaginationButton({
      className: tailwindMerge.twMerge(theme$1.pages.selector.base, currentPage === page && theme$1.pages.selector.active),
      active: page === currentPage,
      onClick: () => onPageChange(page),
      children: page
    }) }, page)),
    /* @__PURE__ */ jsxRuntime.jsx("li", { children: /* @__PURE__ */ jsxRuntime.jsxs(
      PaginationButton.PaginationNavigation,
      {
        className: tailwindMerge.twMerge(theme$1.pages.next.base, showIcon && theme$1.pages.showIcon),
        onClick: goToNextPage,
        disabled: currentPage === totalPages,
        children: [
          nextLabel,
          showIcon && /* @__PURE__ */ jsxRuntime.jsx(chevronRightIcon.ChevronRightIcon, { "aria-hidden": true, className: theme$1.pages.next.icon })
        ]
      }
    ) })
  ] }) });
});
const TablePagination = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.paginationTheme, provider$1.theme?.pagination, props.theme],
    [get.get(provider$1.clearTheme, "pagination"), props.clearTheme],
    [get.get(provider$1.applyTheme, "pagination"), props.applyTheme]
  );
  const {
    className,
    currentPage,
    nextLabel = "Next",
    onPageChange,
    previousLabel = "Previous",
    showIcons: showIcon = false,
    itemsPerPage,
    totalItems,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.pagination);
  if (!Number.isInteger(currentPage) || currentPage < 1) {
    throw new Error("Invalid props: currentPage must be a positive integer");
  }
  if (!Number.isInteger(itemsPerPage) || itemsPerPage < 1) {
    throw new Error("Invalid props: itemsPerPage must be a positive integer");
  }
  if (!Number.isInteger(totalItems) || totalItems < 0) {
    throw new Error("Invalid props: totalItems must be a non-negative integer");
  }
  const totalPages = totalItems > 0 ? Math.ceil(totalItems / itemsPerPage) : 1;
  const offset = (currentPage - 1) * itemsPerPage;
  const firstItem = totalItems > 0 ? offset + 1 : 0;
  const lastItem = currentPage === totalPages ? totalItems : offset + itemsPerPage;
  function goToNextPage() {
    onPageChange(Math.min(currentPage + 1, totalPages));
  }
  function goToPreviousPage() {
    onPageChange(Math.max(currentPage - 1, 1));
  }
  return /* @__PURE__ */ jsxRuntime.jsxs("nav", { ref, className: tailwindMerge.twMerge(theme$1.base, className), ...restProps, children: [
    /* @__PURE__ */ jsxRuntime.jsxs("div", { role: "status", "aria-live": "polite", "aria-label": "Table Pagination", className: theme$1.layout.table.base, children: [
      "Showing ",
      /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.layout.table.span, children: firstItem }),
      " to\xA0",
      /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.layout.table.span, children: lastItem }),
      " of\xA0",
      /* @__PURE__ */ jsxRuntime.jsx("span", { className: theme$1.layout.table.span, children: totalItems }),
      " Entries"
    ] }),
    /* @__PURE__ */ jsxRuntime.jsxs("ul", { className: theme$1.pages.base, children: [
      /* @__PURE__ */ jsxRuntime.jsx("li", { children: /* @__PURE__ */ jsxRuntime.jsxs(
        PaginationButton.PaginationNavigation,
        {
          className: tailwindMerge.twMerge(theme$1.pages.previous.base, showIcon && theme$1.pages.showIcon),
          onClick: goToPreviousPage,
          disabled: currentPage === 1,
          children: [
            showIcon && /* @__PURE__ */ jsxRuntime.jsx(chevronLeftIcon.ChevronLeftIcon, { "aria-hidden": true, className: theme$1.pages.previous.icon }),
            previousLabel
          ]
        }
      ) }),
      /* @__PURE__ */ jsxRuntime.jsx("li", { children: /* @__PURE__ */ jsxRuntime.jsxs(
        PaginationButton.PaginationNavigation,
        {
          className: tailwindMerge.twMerge(theme$1.pages.next.base, showIcon && theme$1.pages.showIcon),
          onClick: goToNextPage,
          disabled: currentPage === totalPages,
          children: [
            nextLabel,
            showIcon && /* @__PURE__ */ jsxRuntime.jsx(chevronRightIcon.ChevronRightIcon, { "aria-hidden": true, className: theme$1.pages.next.icon })
          ]
        }
      ) })
    ] })
  ] });
});
Pagination.displayName = "Pagination";

exports.Pagination = Pagination;
//# sourceMappingURL=Pagination.cjs.map
