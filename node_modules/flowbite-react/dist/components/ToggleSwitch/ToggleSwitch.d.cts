import type { ComponentProps } from "react";
import type { DynamicStringEnumKeysOf, FlowbiteBoolean, FlowbiteColors, ThemingProps } from "../../types";
import type { TextInputSizes } from "../TextInput";
export interface ToggleSwitchTheme {
    root: ToggleSwitchRootTheme;
    toggle: ToggleSwitchToggleTheme;
}
export interface ToggleSwitchRootTheme {
    base: string;
    active: FlowbiteBoolean;
    label: string;
    input: string;
}
export interface ToggleSwitchToggleTheme {
    base: string;
    sizes: TextInputSizes;
    checked: FlowbiteBoolean & {
        color: FlowbiteColors;
    };
}
export type ToggleSwitchProps = Omit<ComponentProps<"button">, "onChange" | "ref"> & {
    checked: boolean;
    color?: DynamicStringEnumKeysOf<FlowbiteColors>;
    sizing?: DynamicStringEnumKeysOf<TextInputSizes>;
    label?: string;
    onChange: (checked: boolean) => void;
} & ThemingProps<ToggleSwitchTheme>;
export declare const ToggleSwitch: import("react").ForwardRefExoticComponent<Omit<import("react").DetailedHTMLProps<import("react").ButtonHTMLAttributes<HTMLButtonElement>, HTMLButtonElement>, "onChange" | "ref"> & {
    checked: boolean;
    color?: DynamicStringEnumKeysOf<FlowbiteColors>;
    sizing?: DynamicStringEnumKeysOf<TextInputSizes>;
    label?: string;
    onChange: (checked: boolean) => void;
} & ThemingProps<ToggleSwitchTheme> & import("react").RefAttributes<HTMLInputElement>>;
