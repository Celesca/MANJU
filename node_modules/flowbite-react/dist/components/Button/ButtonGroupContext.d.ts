import type { ButtonProps } from "./Button";
export type ButtonGroupContextValue = Pick<ButtonProps, "outline" | "pill"> | undefined;
export declare const ButtonGroupContext: import("react").Context<ButtonGroupContextValue>;
export declare function useButtonGroupContext(): ButtonGroupContextValue;
