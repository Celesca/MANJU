'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var tailwindMerge = require('../../../helpers/tailwind-merge.cjs');
var DatepickerContext = require('../DatepickerContext.cjs');
var helpers = require('../helpers.cjs');

function DatepickerViewsMonth() {
  const {
    theme: rootTheme,
    minDate,
    maxDate,
    filterDate,
    selectedDate,
    viewDate,
    language,
    setViewDate,
    setView
  } = DatepickerContext.useDatePickerContext();
  const theme = rootTheme.views.months;
  return /* @__PURE__ */ jsxRuntime.jsx("div", { className: theme.items.base, children: [...Array(12)].map((_month, index) => {
    const newDate = /* @__PURE__ */ new Date();
    newDate.setMonth(index, 1);
    newDate.setFullYear(viewDate.getFullYear());
    const month = helpers.getFormattedDate(language, newDate, { month: "short" });
    const isSelected = selectedDate && helpers.isDateEqual(selectedDate, newDate);
    const isDisabled = !helpers.isDateInRange(newDate, minDate, maxDate) || filterDate && !filterDate(newDate, helpers.Views.Months);
    return /* @__PURE__ */ jsxRuntime.jsx(
      "button",
      {
        disabled: isDisabled,
        type: "button",
        className: tailwindMerge.twMerge(
          theme.items.item.base,
          isSelected && theme.items.item.selected,
          isDisabled && theme.items.item.disabled
        ),
        onClick: () => {
          if (isDisabled) return;
          setViewDate(newDate);
          setView(helpers.Views.Days);
        },
        children: month
      },
      index
    );
  }) });
}
DatepickerViewsMonth.displayName = "DatepickerViewsMonth";

exports.DatepickerViewsMonth = DatepickerViewsMonth;
//# sourceMappingURL=Months.cjs.map
