import type { ThemingProps } from "../../types";
import { type TextInputProps, type TextInputTheme } from "../TextInput";
import { Views, WeekStart } from "./helpers";
import type { DatepickerViewsDaysTheme } from "./Views/Days";
import { type DatepickerViewsDecadesTheme } from "./Views/Decades";
import { type DatepickerViewsMonthsTheme } from "./Views/Months";
import { type DatepickerViewsYearsTheme } from "./Views/Years";
export interface DatepickerTheme {
    root: {
        base: string;
        input?: TextInputTheme;
    };
    popup: DatepickerPopupTheme;
    views: {
        days: DatepickerViewsDaysTheme;
        months: DatepickerViewsMonthsTheme;
        years: DatepickerViewsYearsTheme;
        decades: DatepickerViewsDecadesTheme;
    };
}
export interface DatepickerPopupTheme {
    root: {
        base: string;
        inline: string;
        inner: string;
    };
    header: {
        base: string;
        title: string;
        selectors: {
            base: string;
            button: {
                base: string;
                prev: string;
                next: string;
                view: string;
            };
        };
    };
    view: {
        base: string;
    };
    footer: {
        base: string;
        button: {
            base: string;
            today: string;
            clear: string;
        };
    };
}
export interface DatepickerRef {
    /**
     * Focus the datepicker input.
     */
    focus: () => void;
    /**
     * Clears the datepicker value back to the defaultValue.
     */
    clear: () => void;
}
export interface DatepickerProps extends Omit<TextInputProps, keyof ThemingProps<DatepickerTheme> | "onChange" | "value" | "defaultValue">, ThemingProps<DatepickerTheme> {
    defaultValue?: Date;
    open?: boolean;
    inline?: boolean;
    autoHide?: boolean;
    showClearButton?: boolean;
    labelClearButton?: string;
    showTodayButton?: boolean;
    labelTodayButton?: string;
    minDate?: Date;
    maxDate?: Date;
    filterDate?: (date: Date, view: Views) => boolean;
    language?: string;
    weekStart?: WeekStart;
    onChange?: (date: Date | null) => void;
    value?: Date | null;
    label?: string;
}
export declare const Datepicker: import("react").ForwardRefExoticComponent<DatepickerProps & import("react").RefAttributes<DatepickerRef>>;
