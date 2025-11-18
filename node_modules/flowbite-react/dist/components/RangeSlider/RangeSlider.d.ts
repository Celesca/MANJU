import type { ComponentProps } from "react";
import type { DynamicStringEnumKeysOf, ThemingProps } from "../../types";
import type { TextInputSizes } from "../TextInput";
export interface RangeSliderTheme {
    root: RangeSliderRootTheme;
    field: RangeSliderFieldTheme;
}
export interface RangeSliderRootTheme {
    base: string;
}
export interface RangeSliderFieldTheme {
    base: string;
    input: {
        base: string;
        sizes: TextInputSizes;
    };
}
export interface RangeSliderProps extends Omit<ComponentProps<"input">, "ref" | "type">, ThemingProps<RangeSliderTheme> {
    sizing?: DynamicStringEnumKeysOf<TextInputSizes>;
}
export declare const RangeSlider: import("react").ForwardRefExoticComponent<RangeSliderProps & import("react").RefAttributes<HTMLInputElement>>;
